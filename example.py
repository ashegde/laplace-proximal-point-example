import numpy as np
import matplotlib.pyplot as plt
import os
from lpp import LaplaceProximalPoint
import argparse

def rastrigin_func(x: np.ndarray) -> float:
    '''
    Rastrigin function: https://en.wikipedia.org/wiki/Rastrigin_function

    Inputs:
    x is (n,)

    Outputs:
    f is a float
    '''
    n = x.shape[0]
    return 10*n + (x**2 - 10*np.sin(2*np.pi*x)).sum()

def main(args: argparse):
    np.random.seed(seed=2024)

    a = 5.12 # rastrigin function prior [-a,a] for each variable
    x0 = a*np.random.rand(2) 
    optim = LaplaceProximalPoint(rastrigin_func, x0, args.lambd, args.delta, args.N)
    xvals = [x0]
    fvals = [rastrigin_func(x0)]

    for ii in range(args.iters):
        xnew = optim.step()
        fnew = rastrigin_func(xnew)
        xvals.append(xnew)
        fvals.append(fnew)

        decay = (fvals[-1] - fvals[-2]) / fvals[-2] 

        if args.disp and ii%args.skip == 0:
            print(f'iter: {ii} | fval: {fnew:0.5f} | percent change: {decay:0.3f}')

    if args.plot:
        figdir = 'figs'
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        
        # plot loss
        fig, ax = plt.subplots()
        ax.plot(fvals, 'r')
        ax.set_yscale('log')
        ax.set_xlabel(f'Iteration')
        ax.set_ylabel(f"Loss value")
        ax.set_title(f"Loss")
        plt.savefig(f"{figdir}/loss.{args.ext}",dpi=300)
        plt.close()

        # plot solution path
        fig, ax = plt.subplots()
        x1_ = np.linspace(-a, a, num=200)
        x2_ = np.linspace(-a, a, num=200)
        x1,x2 = np.meshgrid(x1_, x2_)
        lvls = np.apply_along_axis(rastrigin_func, -1, np.concatenate((x1[...,None], x2[...,None]), axis=-1))
        c=ax.contourf(x1, x2, lvls, 50)
        ax.plot([x[0] for x in xvals], [x[1] for x in xvals],'r')
        ax.plot(xvals[0][0], xvals[0][1], 'sr')
        ax.plot(xvals[-1][0], xvals[-1][1],'*w')
        ax.set_title(f'Optimization path')
        plt.colorbar(c)
        plt.savefig(f"{figdir}/solpath.{args.ext}",dpi=300)
        plt.close()

        
        

# define an example objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run example to illustrate the Laplace Proximal Point algorithm')
    parser.add_argument('--lambd', type=float, default=1.0, help='proximal operator parameter')
    parser.add_argument('--delta', type=float, default=1.0, help='temperature parameter')
    parser.add_argument('--N', type=int, default=50, help='number of samples per step')
    parser.add_argument('--iters', type=int, default=1000, help='maximum number of iterations')
    parser.add_argument('--plot', type=bool, default=True, help='plot results')
    parser.add_argument('--ext', type=str, default='png', help='plot extension')
    parser.add_argument('--disp', type=bool, default=True, help='display optimization output')
    parser.add_argument('--skip', type=int, default=100, help='display optimization output')
    args = parser.parse_args()
    main(args)