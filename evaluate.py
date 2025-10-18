from argparse import ArgumentParser, Namespace



def main():
    parser = ArgumentParser()
    parser.add_argument("--red-agent", required=True, type=str)
    parser.add_argument("--black-agent", required=True, type=str)
    
    opts = parser.parse_args()

    


if __name__ == '__main__':
    main()
