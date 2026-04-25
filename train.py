import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from WM import VAE, MDNRNN, vae_loss

# --- Self-Driving Simulator Dataset ---

class DrivingSimulationDataset(Dataset):
    def __init__(self, num_sequences=500, seq_len=30, img_size=64):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.img_size = img_size

    def __len__(self):
        return self.num_sequences

    def render_frame(self, car_pos, lane_phase, obj_z, obj_lane):
        # Create a background: Sky and Grass
        frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        frame[:self.img_size//2, :] = [135, 206, 235]       # Sky (Light Blue)
        frame[self.img_size//2:, :] = [34, 139, 34]         # Grass (Green)
        
        # Road (Gray) - Perspective trapezoid
        road_pts = np.array([
            [self.img_size//2 - 2, self.img_size//2],
            [self.img_size//2 + 2, self.img_size//2],
            [self.img_size + 40, self.img_size],
            [-40, self.img_size]
        ])
        road_pts[:, 0] -= int(car_pos * 40)
        cv2.fillPoly(frame, [road_pts], (100, 100, 100))
        
        # Lane lines (White dashes)
        num_dashes = 5
        for i in range(num_dashes):
            z = (i + lane_phase) / num_dashes
            y = int(self.img_size//2 + z * (self.img_size//2))
            x_center = self.img_size // 2 - int(car_pos * 40 * z)
            if y < self.img_size:
                cv2.line(frame, (x_center, y), (x_center, min(y+int(5*z), self.img_size)), (255, 255, 255), max(1, int(2*z)))
        
        # Object (Red Car/Box)
        if obj_z > 0:
            # Perspective mapping for object
            # obj_z goes from 0 (horizon) to 1 (near)
            y = int(self.img_size//2 + obj_z * (self.img_size//2))
            # lane position: -0.5 (left lane), 0.5 (right lane)
            x_center = self.img_size // 2 + int((obj_lane - car_pos) * 40 * obj_z)
            w = int(2 + obj_z * 20)
            h = int(1 + obj_z * 10)
            if y < self.img_size:
                cv2.rectangle(frame, (x_center - w//2, y - h), (x_center + w//2, y), (200, 50, 50), -1)
                
        return frame.transpose(2, 0, 1) / 255.0

    def __getitem__(self, idx):
        car_pos = 0.0
        lane_phase = 0.0
        # Object initial state: far away (z=0) in a random lane
        obj_z = 0.0
        obj_lane = np.random.choice([-0.4, 0.4])
        obj_speed = np.random.uniform(0.02, 0.05)
        
        frames = []
        actions = []
        
        for _ in range(self.seq_len):
            frames.append(self.render_frame(car_pos, lane_phase, obj_z, obj_lane))
            
            # Action: steer left/right
            action = np.random.uniform(-0.15, 0.15, size=2)
            actions.append(action)
            
            # Update state
            car_pos += action[0]
            car_pos = np.clip(car_pos, -1.0, 1.0)
            lane_phase = (lane_phase + 0.1) % 1.0
            
            obj_z += obj_speed
            if obj_z > 1.2: # Reset object if it passes
                obj_z = 0.0
                obj_lane = np.random.choice([-0.4, 0.4])
            
        return torch.tensor(np.array(frames), dtype=torch.float32), torch.tensor(np.array(actions), dtype=torch.float32)

# --- Training Loops ---

def train_vae(vae, dataloader, device, epochs=10):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.to(device)
    vae.train()
    print(f"Starting VAE Training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for frames, _ in dataloader:
            # Flatten sequences for VAE training
            frames = frames.view(-1, 3, 64, 64).to(device)
            
            optimizer.zero_grad()
            _, recon_x, mu, logvar = vae(frames)
            loss = vae_loss(recon_x, frames, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader.dataset):.4f}")

def mdn_loss(pi, mu, sigma, target):
    target = target.unsqueeze(2).expand_as(mu)
    m = torch.distributions.Normal(mu, sigma)
    log_probs = m.log_prob(target)
    log_probs = log_probs.sum(dim=-1)
    log_pi = torch.log(pi + 1e-8)
    loss = -torch.logsumexp(log_pi + log_probs, dim=-1)
    return loss.mean()

def train_rnn(vae, rnn, dataloader, device, epochs=10):
    optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
    vae.to(device)
    vae.eval()
    rnn.to(device)
    rnn.train()
    print(f"\nStarting MDN-RNN Training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for frames, actions in dataloader:
            frames, actions = frames.to(device), actions.to(device)
            with torch.no_grad():
                bs, seq_len, c, h, w = frames.shape
                flat_frames = frames.view(-1, c, h, w)
                mu_z, _ = vae.encode(flat_frames)
                z = mu_z.view(bs, seq_len, -1)
            
            z_input = z[:, :-1, :]
            actions_input = actions[:, :-1, :]
            z_target = z[:, 1:, :]
            
            optimizer.zero_grad()
            pi, mu_pred, sigma_pred, _ = rnn(z_input, actions_input)
            
            loss = mdn_loss(pi, mu_pred, sigma_pred, z_target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# --- Main Execution ---
    # Hyperparams
    Z_DIM = 32
    HIDDEN_DIM = 128
    ACTION_DIM = 2
    NUM_GAUSSIANS = 5
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    dataset = DrivingSimulationDataset(num_sequences=2000, seq_len=30)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    vae = VAE(z_dim=Z_DIM).to(DEVICE)
    rnn = MDNRNN(z_dim=Z_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS).to(DEVICE)
    
    # Load if exists
    if os.path.exists("vae.pth") and os.path.exists("rnn.pth"):
        print("Loading existing models...")
        vae.load_state_dict(torch.load("vae.pth"))
        rnn.load_state_dict(torch.load("rnn.pth"))
    else:
        # Train
        train_vae(vae, dataloader, DEVICE, epochs=15)
        train_rnn(vae, rnn, dataloader, DEVICE, epochs=15)
        torch.save(vae.state_dict(), "vae.pth")
        torch.save(rnn.state_dict(), "rnn.pth")
    
    # --- 4K Imagination Step ---
    print("\nGenerating 4K Imagination Video (imagination_4k.mp4)...")
    vae.eval()
    rnn.eval()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 4K resolution
    WIDTH, HEIGHT = 3840, 2160
    out = cv2.VideoWriter('imagination_4k.mp4', fourcc, 15, (WIDTH, HEIGHT))
    
    with torch.no_grad():
        z = torch.zeros(1, 1, Z_DIM).to(DEVICE)
        hidden = None
        
        for i in range(150):
            steering = 0.2 * np.sin(i / 8.0)
            action = torch.tensor([[[steering, 0.0]]], dtype=torch.float32).to(DEVICE)
            
            pi, mu, sigma, hidden = rnn(z, action, hidden)
            mode = torch.argmax(pi, dim=-1)
            z = mu[0, 0, mode[0, 0]].unsqueeze(0).unsqueeze(0)
            
            img = vae.decode(z.squeeze(0)).squeeze(0)
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # High-quality upscale to 4K
            img_4k = cv2.resize(img_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            
            # Add some "Self-Driving" UI overlay for 4K feel
            cv2.putText(img_4k, f"WORLD MODEL IMAGINATION | STEER: {steering:.2f}", (100, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(img_4k, "LATENT SPACE: 32D | DEVICE: CUDA", (100, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            out.write(img_4k)
            
    out.release()
    print("4K Video saved as 'imagination_4k.mp4'!")
    print("Pipeline Complete!")
