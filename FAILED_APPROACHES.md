# FAILED APPROACHES - read before touching training code, do NOT retry these

2026-07-15 | forward PINN stuck at relL2 ~0.29 | soft IC/BC penalties + equal-ish manual weights (w_bc=w_ic=10) | network collapses to a low-amplitude solution (predicted peak ~34C vs true 54.6C) and pde loss plateaus at ~0.5, L-BFGS with 800 iters does NOT escape the basin | remove the competing loss terms entirely: hard-constrain IC with a time envelope and hard-constrain Neumann BC with cosine input features so only the PDE residual remains

2026-07-15 | same plateau | gradient-norm adaptive weighting (Wang et al. style, recomputed every 500 epochs) | BC/IC terms converge fast so their grad norms shrink, the scheme then hands them HUGE weights which locks the network at the trivial near-ambient solution even harder | if adaptive weighting is ever revisited, cap the weights and exclude already-converged terms

2026-07-15 | hard-IC envelope alone (exp2, tau=15) | still plateaued at pde ~0.48, peak ~34 | envelope fixes IC but the localized Gaussian source still underfits with plain (x,t) inputs - spectral bias | combine envelope WITH cosine Fourier features in x, not either alone

2026-07-15 | running 3 torch training jobs in parallel on this container | CPU thread thrashing, every job froze at epoch 0 for 10+ min, all diagnostics timed out | run ONE training job at a time with OMP_NUM_THREADS=4 and torch.set_num_threads(4)
