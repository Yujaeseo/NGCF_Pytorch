import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.optim as optim
import horovod.torch as hvd
from time import time

from util.parser import parse_args
from util.load_data import Data
from util.eval_model import test_model
from NGCF import NGCF

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(args.device) + " for computations")

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    train_file = args.data_path + '/' + args.dataset + '/' + args.train_file
    test_file = args.data_path + '/' + args.dataset + '/' + args.test_file
    file_path = args.data_path + '/' + args.dataset

    data = Data(file_path, train_file, test_file, args.batch_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        data, shuffle=False ,num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(
        data,
        batch_size = args.batch_size,
        sampler = train_sampler
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
    model = NGCF(data.n_users, data.n_items, norm_adj, args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())
    
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(),compression = compression)
    
    start_epoch = 0
    total_time = 0
    model.train()
    for epoch in range(start_epoch, args.epoch):
        t0_start = time()
        loss = 0

        for idx, (users, pos_items, neg_items) in enumerate(train_loader):
            optimizer.zero_grad()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items,
                                                                        drop_flag=args.node_dropout)
                                                                        
            batch_loss = model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

            batch_loss.backward()
            optimizer.step() 

            loss += batch_loss

        t0_end = time()
        
        loss = hvd.allreduce(loss, "loss").item()
        run_time = hvd.allreduce(torch.tensor(t0_end - t0_start), "time").item()
        
        if hvd.rank() == 0:
            print('epoch {} : loss {} , time {}s'.format(epoch + 1, loss, run_time))
            total_time += run_time


        if (epoch + 1) % 20 == 0:
            data.set_mode(2)
            ret = test_model(test_loader, data, model, args.batch_size ,eval(args.ks) ,drop_flag=False)
            data.set_mode(1)
            
            recall = hvd.allreduce(torch.tensor(ret['recall'][0]), "recall").item()
            ngcf = hvd.allreduce(torch.tensor(ret['ndcg'][0]), "ndcg").item()

            if hvd.rank() == 0:
                print("Recall :" + str(recall))
                print("NGCF : " + str(ngcf))

    print("Total run time :" + str(total_time))