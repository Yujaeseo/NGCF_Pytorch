import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.optim as optim
from time import time

from util.parser import parse_args
from util.load_data import Data
from util.eval_model import test_model

from NGCF import NGCF

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(args.device) + " for computations")

    train_file = args.data_path + '/' + args.dataset + '/' + args.train_file
    test_file = args.data_path + '/' + args.dataset + '/' + args.test_file
    file_path = args.data_path + '/' + args.dataset
    
    data = Data(file_path, train_file, test_file, args.batch_size)

    train_loader = DataLoader(
        data,
        batch_size = args.batch_size,
        sampler = SequentialSampler(data),
        num_workers = 8
    )

    test_loader = DataLoader(
        data,
        batch_size = args.batch_size,
        sampler = SequentialSampler(data),
        num_workers = 8
    )

    args.node_dropout = eval(args.node_dropout)
    args.message_dropout = eval(args.message_dropout)

    norm_adj = data.get_adj_mat()
    model = NGCF(data.n_users, data.n_items, norm_adj, args).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.resume != 0:
        checkpoint_info = torch.load(f'./Checkpoint_mod/{args.checkpoint_prefix}_epoch {args.resume}.pth')
        model.load_state_dict(checkpoint_info['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        start_epoch = checkpoint_info['epoch'] + 1
    
    total_time = 0

    for epoch in range(start_epoch, args.epoch):
        t0_start = time()
        loss = 0

        for idx, (users, pos_items, neg_items) in enumerate(train_loader):
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items,
                                                                        drop_flag=args.node_dropout)

            batch_loss = model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        t0_end = time()
        print('epoch {} : loss {} , time {}s'.format(epoch + 1, loss.item(), t0_end - t0_start))
        total_time += t0_end-t0_start

        if (epoch + 1) % 20 == 0:
            data.set_mode(2)
            ret = test_model(test_loader, data, model, args.batch_size ,eval(args.ks) ,drop_flag=False)
            data.set_mode(1)
            print(ret)

    print("Total run time :" + str(total_time))




