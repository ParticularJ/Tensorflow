import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1, help="increase output verbosity")
parser.add_argument("-s", "--square", type=int, nargs = "?", default = 3, help="the square caculation")
#parser.add_argument("x", type=int, help="base")
args = parser.parse_args()
answer = args.square ** 2
if args.verbosity==0:
    print(answer)
elif args.verbosity==1:
    print("the square of {} equal {}".format(args.square, answer))
else:
    print("the {}^{}=={}".format(args.square, args.square, answer))

