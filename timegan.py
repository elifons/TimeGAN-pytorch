import torch
from torch import optim
# from torch.autograd.variable import Variable
# from torchvision im port transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from utils.data_utils import MinMaxScaler
from utils.models import *


def timegan(ori_data, parameters):
    """Train TimeGAN and generates new data. 
        
    Args:
        - ori_data: original time-series data
        - parameters: TimeGAN network parameters
    Returns:
        - generated_data: generated time-series data
    """

    # Check if cuda is available and set device
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print(torch.cuda.current_device())
    else:
        device = torch.device("cpu")


    # timegan parameters
    hidden_dim   = parameters['hidden_dim']
    num_layers   = parameters['num_layer']
    num_epochs   = parameters['num_epochs']
    batch_size   = parameters['batch_size']
    z_dim        = dim
    gamma        = 1


    # Normalize data and create dataloader
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    train_loader = DataLoader(torch.from_numpy(ori_data).float(), shuffle=True, batch_size=batch_size, drop_last=True)



    # Build networks
    embedder = EmbedderNet(input_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=hidden_dim)
    embedder.to(device)
    recovery = RecoveryNet(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=3, output_dim=dim)
    recovery.to(device)
    supervisor = SupervisorNet(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers-1, output_dim=hidden_dim)
    supervisor.to(device)
    generator = GeneratorNet(input_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=hidden_dim)
    generator.to(device)
    discriminator = DiscriminatorNet(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    discriminator.to(device)





    # params_e_r = list(embedder.parameters()) + list(recovery.parameters())
    # params_g_s = list(generator.parameters()) + list(supervisor.parameters())

    # E0_optimizer = optim.Adam(params_e_r)
    # GS_optimizer = optim.Adam(params_g_s)

    # G_optimizer = optim.Adam(params_g_s)
    # E_optimizer = optim.Adam(params_e_r)
    # D_optimizer = optim.Adam(discriminator.parameters())


    BCEloss = nn.BCELoss()
    MSEloss = nn.MSELoss()

    # 1. Train Embedding network
    E0_optimizer = optim.Adam(params_e_r)
    params_e_r = list(embedder.parameters()) + list(recovery.parameters())


    step_e_loss = []
    for epoch in range(num_epochs):
        for X in train_loader:
            E0_optimizer.zero_grad()
            X = X.to(device)
            H = embedder(X)
            X_tilde = recovery(H)
            E_loss_T0 = MSEloss(X,X_tilde)
            E_loss0 = 10*torch.sqrt(E_loss_T0)
            E_loss0.backward()
            E0_optimizer.step()
            E0_optimizer.step()
            step_e_loss.append(E_loss_T0.item())
        if epoch % 10 == 0:
            torch.save(embedder.state_dict(), 'checkpoint_embedder_'+str(epoch)+'.pt')
            torch.save(recovery.state_dict(), 'checkpoint_recovery_'+str(epoch)+'.pt')
            print(epoch, np.round(np.sqrt(E_loss_T0.item()),4))


    # 2. Train with supervised loss
    params_g_s = list(generator.parameters()) + list(supervisor.parameters())
    GS_optimizer = optim.Adam(params_g_s)


    step_g_loss_s = []
    for epoch in range(num_epochs):
        for X in train_loader:
            GS_optimizer.zero_grad()
            X = X.to(device)
            H = embedder(X)
            H_hat_supervise = supervisor(H)
            G_loss_S = MSEloss(H[:,1:,:], H_hat_supervise[:,:-1,:])
            G_loss_S.backward()
            GS_optimizer.step()
            step_g_loss_s.append(G_loss_S.item())
        if epoch % 10 == 0:
            torch.save(supervisor.state_dict(), 'checkpoint_supervisor_'+str(epoch)+'.pt')
            torch.save(generator.state_dict(), 'checkpoint_generator_'+str(epoch)+'.pt')
            print('Epoch', epoch, np.round(np.sqrt(G_loss_S.item()),4))


    # 3. Joint training
    G_optimizer = optim.Adam(params_g_s)
    E_optimizer = optim.Adam(params_e_r)
    D_optimizer = optim.Adam(discriminator.parameters())

    step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0, step_d_loss = [], [], [], [], []
    for epoch in range(num_epochs):
        for kk in range(2):
            for X in train_loader:
                G_optimizer.zero_grad()
                E_optimizer.zero_grad()

                X = X.to(device)
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                Z_mb = torch.from_numpy(np.array(Z_mb)).float()
                E_hat = generator(Z_mb)
                H_hat = supervisor(E_hat)
                X_hat = recovery(H_hat)
                Y_fake = discriminator(H_hat)
                Y_fake_e = discriminator(E_hat)

                # Train generator
                G_loss_U = BCEloss(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_e = BCEloss(Y_fake_e, torch.ones_like(Y_fake_e))
                
                H = embedder(X)
                H_hat_supervise = supervisor(H)

                G_loss_S = MSEloss(H[:,1:,:], H_hat_supervise[:,:-1,:])
                
                G_loss_V1 = torch.mean(torch.abs((torch.std(X_hat,0) + 1e-6) - (torch.std(X,0) + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs((torch.mean(X_hat,0)) - (torch.mean(X,0))))
                G_loss_V = G_loss_V1 + G_loss_V2
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
                
                G_loss.backward(retain_graph=True)


                # Train embedder
                X_tilde = recovery(H)
                E_loss_T0 = MSEloss(real_data, X_tilde)
                E_loss0 = 10*torch.sqrt(E_loss_T0)
                E_loss = E_loss0  + 0.1*G_loss_S

                E_loss.backward()
        
                G_optimizer.step()
                E_optimizer.step()
                step_g_loss_u.append(G_loss_U)
                step_g_loss_s.append(G_loss_S)
                step_g_loss_v.append(G_loss_V)
                step_e_loss_t0.append(E_loss_T0)

            
        # Train discriminator
        for X in train_loader:
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb = torch.from_numpy(np.array(Z_mb)).float()
            D_optimizer.zero_grad()

            X = X.to(device)            
            H = embedder(X)
            E_hat = generator(Z_mb).detach()
            H_hat = supervisor(E_hat)
            Y_real = discriminator(H)
            Y_fake = discriminator(H_hat)
            Y_fake_e = discriminator(E_hat)
            
            D_loss_real = BCEloss(Y_real, torch.ones_like(Y_real))
            D_loss_fake = BCEloss(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = BCEloss(Y_fake_e, torch.zeros_like(Y_fake_e))
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

            if D_loss > 0.15:        
                D_loss.backward()
                D_optimizer.step()
            step_d_loss.append(D_loss)

        if epoch % 10 == 0:
            torch.save(supervisor.state_dict(), 'checkpoint_supervisor_'+str(epoch)+'.pt')
            torch.save(generator.state_dict(), 'checkpoint_generator_'+str(epoch)+'.pt')
            torch.save(embedder.state_dict(), 'checkpoint_embedder_'+str(epoch)+'.pt')
            torch.save(recovery.state_dict(), 'checkpoint_recovery_'+str(epoch)+'.pt')
            torch.save(discriminator.state_dict(), 'checkpoint_discriminator_'+str(epoch)+'.pt')
            print('Epoch:', epoch, 'd_loss:', np.round(D_loss.item(),4), 'g_loss_u:', np.round(G_loss_U.item(),4), \
                  'g_loss_s:', np.round(np.sqrt(G_loss_S.item()),4), \
                  'g_loss_v:', np.round(G_loss_V.item(),4), \
                  'e_loss_t0:', np.round(np.sqrt(E_loss_T0.item()),4))


    generator.eval()
    supervisor.eval()
    recovery.eval()

    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Z_mb = torch.from_numpy(np.array(Z_mb)).float().to(device)
    E_hat = generator(Z_mb)
    H_hat = supervisor(E_hat)
    X_hat = recovery(H_hat)
    X_hat = X_hat.detach().cpu()
    generated_data = X_hat * max_val
    generated_data = generated_data + min_val
    
    return generated_data

# TODO:
    # - check losses are correct (same magnitude to original)
    # - check if we need to do zero_grad to the networks
    # - add comments to the training loops
    #  
