from settings import worlds, rsa_models
import argparse

if __name__ == '__main__':
    # Get user input for RSA parameters
    parser = argparse.ArgumentParser("RSA model parameters")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Enable verbose mode")
    parser.add_argument("--RSA", action="store", type=str, default="classic_RSA", help="RSA model: 'classic_RSA', 'lexical_uncertainty_RSA'")
    parser.add_argument("--world", action="store", type=str, default="bergen_2016_fig1", help="World initialization: 'bergen_2016_fig1', 'bergen_2016_fig3', 'degen_2023_fig1a', 'degen_2023_fig1e', 'bergen_2016_fig6'")
    parser.add_argument("--alpha", action="store", type=float, default=1, help="Speaker pragmatism parameter of the RSA model")
    parser.add_argument("--depth", action="store", type=int, default=1, help="Number of speaker/listener inferences of the RSA model")
    args = parser.parse_args()

    # Initialize the RSA model with verbose mode
    rsa_model = rsa_models[args.RSA]
    world = worlds[args.world]
    rsa = rsa_model(world, save_directory='papers_experiments/', alpha=args.alpha, depth=args.depth)
    # Run the RSA model
    probabilities = rsa.run(args.verbose)
