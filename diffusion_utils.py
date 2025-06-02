import torch
import math

def make_cosine_schedule(num_timesteps, s=0.008, device='cpu'):
    t = torch.arange(num_timesteps + 1, device=device, dtype=torch.float64)
    # The formula in the notebook is alpha_prod_t = cos^2 ( (t/T + s) / (1+s) * pi/2 )
    # and then normalized by alpha_prod_0.
    # Note: notebook has t / (num_timesteps+1), if t goes up to num_timesteps, then t/num_timesteps
    # Let's use num_timesteps as T in the formula:
    # T = num_timesteps
    # f_t = cos( (t/T + s) / (1+s) * pi/2 )^2
    # alpha_bar_t = f_t / f_0
    
    # Corrected based on common implementations and original DDPM paper discussions related to cosine schedule:
    # The argument to cos is ( (t / num_timesteps) * pi/2 ) or similar for improved schedules
    # The notebook uses: alpha_prod = torch.cos(0.5 * (t / (num_timesteps+1) + s) / (1.0 + s) * math.pi)**2
    # This t goes from 0 to num_timesteps.
    # alpha_prod_t is indexed from 0 to num_timesteps-1.
    # So we generate num_timesteps+1 values, then take [1:] for alpha_prod[0]...alpha_prod[num_timesteps-1]
    # relative to t=1...num_timesteps in formulas.
    
    # The notebook formula:
    # t_steps = torch.arange(num_timesteps + 1, device=device, dtype=torch.float64) # 0 to T
    # f_t = torch.cos(((t_steps / num_timesteps) + s) / (1 + s) * (math.pi / 2))**2 # Using num_timesteps as T
    # alpha_prod = f_t / f_t[0]
    # return alpha_prod[1:].float() # Return for t=1...T (num_timesteps elements)

    # Let's use the exact formula from the notebook for t from 0 to num_timesteps
    t_range = torch.arange(num_timesteps + 1, device=device, dtype=torch.float64) # Tensor from 0 to num_timesteps
    # The notebook calculates alpha_prod for t from 0 to num_timesteps (total T+1 values)
    # Then uses alpha_prod[1:] to get T values.
    # alpha_prod[t-1] in diffusion step means t goes from 1 to T.
    # So indices are 0 to T-1.
    alpha_prod = torch.cos(0.5 * (t_range / num_timesteps + s) / (1.0 + s) * math.pi)**2
    alpha_prod = alpha_prod / alpha_prod[0] # Normalize so alpha_prod_0 = 1
    return alpha_prod[1:].float() # Return T values, for t=1...T


def make_linear_schedule(num_timesteps, start=1e-4, end=2e-2, device='cpu'): # Notebook used 7e-3 for T=1000
    betas = torch.linspace(start, end, num_timesteps, device=device, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_prod = torch.cumprod(alphas, dim=0)
    return alpha_prod.float()

def get_alpha_prod(time_steps, schedule_type="linear", linear_start=1e-4, linear_end=2e-2, cosine_s=0.008, device='cpu'):
    if schedule_type == "linear":
        print(f"Using linear schedule with start={linear_start}, end={linear_end}, steps={time_steps}")
        return make_linear_schedule(time_steps, start=linear_start, end=linear_end, device=device)
    elif schedule_type == "cosine":
        print(f"Using cosine schedule with s={cosine_s}, steps={time_steps}")
        return make_cosine_schedule(time_steps, s=cosine_s, device=device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

if __name__ == '__main__':
    # Test schedules
    T = 1000
    alpha_prod_lin = make_linear_schedule(T, start=1e-4, end=7e-3)
    alpha_prod_cos = make_cosine_schedule(T, s=0.008)

    print("Linear schedule sample (first 5, last 5):")
    print(alpha_prod_lin[:5])
    print(alpha_prod_lin[-5:])
    print("Shape:", alpha_prod_lin.shape)


    print("\nCosine schedule sample (first 5, last 5):")
    print(alpha_prod_cos[:5])
    print(alpha_prod_cos[-5:])
    print("Shape:", alpha_prod_cos.shape)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.plot(alpha_prod_cos.cpu().numpy(), label='cosine (s=0.008)')
    plt.plot(alpha_prod_lin.cpu().numpy(), label='linear (1e-4 to 7e-3)')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\bar{\alpha}_t$')
    plt.title(f'Variance Schedules for T={T}')
    plt.legend()
    plt.show()