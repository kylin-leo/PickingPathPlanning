import numpy as np
import math
import scipy.io as sio

def crossover(seq1, seq2, mask, desired_m):

    newseq1 = np.copy(seq1)
    newseq2 = np.copy(seq2)

    # tmp = mask[:, np.round(seq1)]
    # num = np.sum(tmp, 1)
    # if np.sum(np.abs(num - 1)) > 0:
    #     print("error")
    #
    # tmp = mask[:, np.round(seq2)]
    # num = np.sum(tmp, 1)
    # if np.sum(np.abs(num - 1)) > 0:
    #     print("error")

    lens = len(seq1)
    r = int(np.random.randint(1, lens-1, 1))

    for i in range(0, r):
        ind = np.where(np.squeeze(mask[:, seq2[i]]) > 0 )
        v_ind = np.where(np.squeeze(mask[ind[0],:]) > 0)
        np.random.shuffle(v_ind[0])
        j = 0
        while j < len(v_ind[0]) and v_ind[0][j] not in newseq1[r:lens]:
            j = j + 1

        if j < len(v_ind[0]):
            idx = np.where(newseq1[r:lens] == v_ind[0][j])
            np.random.shuffle(idx)
            newseq1[idx[0][0]+r] = newseq1[i]
            newseq1[i] = v_ind[0][j]
        else:
            k = 0
            while k < len(v_ind[0]) and v_ind[0][k] not in newseq1[0:r]:
                k = k + 1

            if k < len(v_ind[0]):
                idx = np.where(newseq1[0:r] == v_ind[0][k])
                np.random.shuffle(idx)
                newseq1[idx[0][0]] = v_ind[0][k]

    # tmp = np.minimum(mask[:, np.round(newseq1)], np.ones_like(mask[:, np.round(newseq1).astype(int)]))
    # num = np.sum(tmp, 1)
    # if np.sum(np.abs(num - desired_m)) > 0:
    #     print("func error")

    for i in range(r, lens):
        ind = np.where(np.squeeze(mask[:, seq1[i]]) > 0)
        v_ind = np.where(np.squeeze(mask[ind[0],:]) > 0)
        np.random.shuffle(v_ind[0])
        j = 0
        while j < len(v_ind[0]) and v_ind[0][j] not in newseq2[0:r]:
            j = j + 1

        if j < len(v_ind[0]):
            idx = np.where(newseq2[0:r] == v_ind[0][j])
            np.random.shuffle(idx)
            newseq2[idx[0][0]] = newseq2[i]
            newseq2[i] = v_ind[0][j]
        else:
            k = 0
            while k < len(v_ind[0]) and v_ind[0][k] not in newseq2[r:lens]:
                k = k + 1

            if k < len(v_ind[0]):
                idx = np.where(newseq2[r:lens] == v_ind[0][k])
                np.random.shuffle(idx)
                newseq2[idx[0][0]+r] = v_ind[0][k]


    # tmp = np.minimum(mask[:, np.round(newseq2)], np.ones_like(mask[:, np.round(newseq2).astype(int)]))
    # num = np.sum(tmp, 1)
    # if np.sum(np.abs(num - desired_m)) > 0:
    #     print("func error")


    return newseq1, newseq2


def  func_seq_local(seq, mask, Q):
    # neighboring seq
    #print(seq)
    seq_local = np.reshape(np.zeros_like(seq),(-1,1))

    for i in range(seq.shape[0]):

        seq_n = np.reshape(np.copy(seq), (-1, 1))
        ind = np.where(np.squeeze(mask[:, seq_n[i]]) > 0)
        #print(seq_n[i])
        v_ind = np.where(np.squeeze(mask[ind[0],:]) > 0)
        np.random.shuffle(v_ind[0])
        j = 0
        if i > 0 and i < seq.shape[0]-1:
            while j < v_ind[0].shape[0] and Q[v_ind[0][j]+1, 0] != Q[seq_n[i-1]+1,0] and Q[v_ind[0][j]+1, 0] != Q[seq_n[i+1]+1,0]:
                j = j + 1
        j = np.minimum(j, v_ind[0].shape[0]-1)
        seq_n[i] = v_ind[0][j]
        #print(v_ind[0][0])
        seq_local = np.concatenate([seq_local, seq_n], axis = 1)
        #print(seq_n.T)



    roadmap = np.zeros((seq.shape[0], 3),dtype=int)
    j = 0
    roadmap[j, 0] = Q[seq[0] + 1 , 0]
    roadmap[j, 1] = 0
    roadmap[j, 2] = 1
    for i in range(1, seq.shape[0]):
        if Q[seq[i]+1, 0] != Q[seq[i-1]+1, 0]:
            j = j + 1
            roadmap[j, 0] = Q[seq[i]+1, 0]
            roadmap[j, 1] = i
        roadmap[j, 2] = roadmap[j, 2] + 1
    num_road = j + 1



    # for i in range(2*num_road):
    #     seq_n2 = np.reshape(np.copy(seq), (-1, 1))
    #     idx = np.random.randint(0, num_road, 2, dtype = int)
    #     if idx[0] == idx[1]:
    #         if roadmap[idx[1],2] > 1:
    #             idy = np.random.permutation(range(roadmap[idx[1], 1],roadmap[idx[1], 1]+roadmap[idx[1],2]))
    #             #np.random.randint(roadmap[idx[1], 1],roadmap[idx[1], 1]+roadmap[idx[1],2], 2, dtype=int)
    #             seq_n2[[idy[0],idy[1]]] = seq_n2[[idy[1],idy[0]]]
    #     elif idx[0] < idx[1]:
    #         tmp1 =seq_n2[0:roadmap[idx[0],1]]
    #         tmp2 =seq_n2[roadmap[idx[1],1]:roadmap[idx[1],1]+roadmap[idx[1],2]]
    #         tmp3=seq_n2[roadmap[idx[0],1]+roadmap[idx[0],2]:roadmap[idx[1],1]]
    #         tmp4=seq_n2[roadmap[idx[0],1]:roadmap[idx[0],1]+roadmap[idx[0],2]]
    #         tmp5=seq_n2[roadmap[idx[1],1]+roadmap[idx[1],2]::]
    #         if np.mod(idx[1] - idx[0], 2) == 1:
    #             tmp2 = tmp2[::-1]
    #             tmp3 = tmp3[::-1]
    #         seq_n2 = np.concatenate([tmp1, tmp2, tmp3, tmp4, tmp5])
    #     else:
    #         tmp1 =seq_n2[0:roadmap[idx[1],1]]
    #         tmp2 =seq_n2[roadmap[idx[0],1]:roadmap[idx[0],1]+roadmap[idx[0],2]]
    #         tmp3=seq_n2[roadmap[idx[1],1]+roadmap[idx[1],2]:roadmap[idx[0],1]]
    #         tmp4=seq_n2[roadmap[idx[1],1]:roadmap[idx[1],1]+roadmap[idx[1],2]]
    #         tmp5=seq_n2[roadmap[idx[0],1]+roadmap[idx[0],2]::]
    #         if np.mod(idx[0] - idx[1], 2) == 1:
    #             tmp2 = tmp2[::-1]
    #             tmp3 = tmp3[::-1]
    #         seq_n2 = np.concatenate([tmp1, tmp2, tmp3, tmp4, tmp5])

    for i in range(num_road):
        seq_n2 = np.reshape(np.copy(seq), (-1, 1))
        idx = np.random.randint(0, num_road, 1, dtype=int)

        if roadmap[idx[0], 2] > 1:
            idy = np.random.permutation(range(roadmap[idx[0], 1], roadmap[idx[0], 1] + roadmap[idx[0], 2]))
            # np.random.randint(roadmap[idx[1], 1],roadmap[idx[1], 1]+roadmap[idx[1],2], 2, dtype=int)
            seq_n2[[idy[0], idy[1]]] = seq_n2[[idy[1], idy[0]]]
            seq_local = np.concatenate([seq_local, seq_n2], axis=1)

        seq_n3 = np.reshape(np.copy(seq), (-1, 1))
        idx = np.random.randint(0, num_road-1, 1, dtype=int)

        tmp1 = seq_n3[0:roadmap[idx[0], 1]]
        tmp2 = seq_n3[roadmap[idx[0]+1, 1]:roadmap[idx[0]+1, 1] + roadmap[idx[0]+1, 2]]
        tmp3 = seq_n3[roadmap[idx[0], 1]:roadmap[idx[0], 1] + roadmap[idx[0], 2]]
        tmp4 = seq_n3[roadmap[idx[0]+1, 1] + roadmap[idx[0]+1, 2]::]

        tmp2 = tmp2[::-1]
        tmp3 = tmp3[::-1]
        seq_n3 = np.concatenate([tmp1, tmp2, tmp3, tmp4])

    #tmp = np.copy(seq_n2[idx[0]])
    #seq_n2[[idx[0],idx[1]]] = seq_n2[[idx[1],idx[0]]]
    #seq_n2[idx[1]] = tmp
        seq_local = np.concatenate([seq_local, seq_n3], axis=1)

    return seq_local[:,1::]


def costtime_between_points(p1, p2, d, t1, t2, t3, B):

    if p1[0] == p2[0]:
        if d == 0 and p1[1] < p2[1]:
            T = 9999
        elif d == 0 and p1[1] >= p2[1]:
            T = t1 * (p1[1] - p2[1])
        elif d == 1 and p1[1] > p2[1]:
            T = 9999
        else:
            T = t1 * (p2[1] - p1[1])
    else:
        if d == 0:
            T = t1*(p1[1] + 1) + t1*(p2[1] + 1) + t2*np.abs(p1[0] - p2[0]) + t3*np.abs(p1[2]) + t3*np.abs(p2[2])
        else:
            T = t1 * (B - p1[1]) + t1 * (B - p2[1]) + t2*np.abs(p1[0] - p2[0]) + t3*np.abs(p1[2]) + t3*np.abs(p2[2])

    # if p1[0] == p2[0]:
    #     print("same")
    return T

def func_objective(T, seq, Q, mask, desired_m):

    d = 0
    total_time1 = T[0, seq[0]+1, d]
    d = 1 - d
    for i in range(seq.shape[0]-1):

        total_time1 = total_time1 + T[seq[i]+1, seq[i+1]+1, d]
        if Q[seq[i]+1,0] != Q[seq[i+1]+1,0]:
            d = 1 - d
        else:
            continue

    wrongamount = 0
    tmp = np.minimum(mask[:, seq], np.ones_like(mask[:, seq.astype(int)]))
    num = np.sum(tmp, 1)
    if np.sum(np.abs(num - desired_m)) > 0:
        wrongamount = 9999
        print("func error")
    if np.min(num - desired_m) < 0:
        wrongamount = 9999

    tmp_mask = np.copy(np.sum(mask,0))
    sparsity = 1
    for i in range(seq.shape[0]):
        tmp_mask[seq[i]]  = tmp_mask[seq[i]] - 1

    emptyposition = np.where(tmp_mask==0)
    if emptyposition[0].size > 0:
        sparsity = -50 * emptyposition[0].size

    runout = 0
    runoutposition = np.where(tmp_mask < 0)
    if runoutposition[0].size > 0:
        runout = 9999
    # d = 1
    # total_time2 = T[0, seq[0]+1, d]
    # d = 1 - d
    # for i in range(seq.shape[0]-1):
    #     if Q[seq[i]+1,0] != Q[seq[i+1]+1,0]:
    #         d = 1 - d
    #     else:
    #         continue
    #     total_time2 = total_time2 + T[seq[i]+1, seq[i+1]+1, d]


    # if total_time1 < total_time2:
    #     return total_time1
    # else:
    #     return total_time2
    return total_time1 + wrongamount + runout + sparsity

def func_diff_evolution(data, prob, num_generation):

    #desired = data.desired
    mask = data['mask']
    T = data['T']
    num_box = data['num_box']
    Q = data['Q']
    desired_m = np.copy(data['desired_amount'])
    LARGE_VALUE = 99999

    num_product_type = mask.shape[0]
    num_position = mask.shape[1]
    num_population = 1 * math.ceil(math.pow(math.perm(num_position, num_product_type) , (1 / 3)))

    print("number of population:" + str(num_population))

    seq_population = - np.ones((num_population, num_box),dtype = int)
    # population
    #initialization
    desired_type = np.zeros((num_box), dtype = int)
    j = 0
    for i in range(num_box):
        if desired_m[j] > 0:
            desired_type[i] = j
            desired_m[j] = desired_m[j] - 1
        else:
            j = j + 1
            desired_type[i] = j
            desired_m[j] = desired_m[j] - 1

    desired_m = np.copy(data['desired_amount'])

    for i in range(num_population):
        product = np.zeros((num_box))
        desired_type_cur = np.copy(desired_type)
        np.random.shuffle(desired_type_cur)
        tmp_mask = np.copy(mask)
        for j in range(num_box):
            ind = np.where(tmp_mask[desired_type_cur[j],:] > 0)
            product[j] = ind[0][np.random.choice(ind[0].size, 1)]
            tmp_mask[:, product[j].astype(int)] = tmp_mask[:, product[j].astype(int)] - 1
        np.random.shuffle(product)

        roadway = np.zeros((num_box))
        for j in range(num_box):
            roadway[j] = Q[product[j].astype(int)+1, 0]
        order =np.argsort(roadway)
        seq_population[i, :] = product[order]
        # tmp = np.minimum(mask[:, product.astype(int)],np.ones_like(mask[:, product.astype(int)]))
        # num = np.sum(tmp, 1)
        # if np.sum(np.abs(num - desired_m)) > 0:
        #     print("error")



        #print(num)
    #select_ratio = 0.8
    cross_ratio = 0.5

    bestrms = np.zeros(num_generation)
    for g in range(num_generation):

        seq_population_g = seq_population
        rms = np.zeros((num_population))
        for i in range(num_population):
            seq_cur = seq_population[i,:]
            rms[i] = func_objective(data['T'], seq_cur, Q, data['mask'], data['desired_amount'])

        print(str(g) + " th best:" + str(np.min(rms)))
        bestrms[g] = np.min(rms)

        select_flag = np.random.rand(num_population)*2 >  rms/np.mean(rms)

        for i in range(num_population):

            if select_flag[i] == False and np.random.rand(1) < cross_ratio:
                p = np.random.permutation(num_population)

                # tmp = np.minimum(mask[:, seq_population_g[p[0]].astype(int)], np.ones_like(mask[:, seq_population_g[p[0]].astype(int)]))
                # num = np.sum(tmp, 1)
                # if np.sum(np.abs(num - desired_m)) > 0:
                #     print("error")
                # tmp = np.minimum(mask[:, seq_population_g[p[1]].astype(int)], np.ones_like(mask[:, seq_population_g[p[0]].astype(int)]))
                # num = np.sum(tmp, 1)
                # if np.sum(np.abs(num - desired_m)) > 0:
                #     print("error")
                newseq1, newseq2 = crossover(seq_population_g[p[0], :], seq_population_g[p[1], :], mask, desired_m)
                if func_objective(data['T'], newseq1,Q, data['mask'], data['desired_amount'])\
                        < func_objective(data['T'], newseq2,Q, data['mask'], data['desired_amount']):
                    seq_population[i,:] = newseq1
                else:
                    seq_population[i, :] = newseq2
                #select_flag[i] = True
                #product = seq_population[i, :]
                # tmp = np.minimum(mask[:, np.round(product)],
                #                      np.ones_like(mask[:, np.round(product)]))
                # num = np.sum(tmp, 1)
                # if np.sum(np.abs(num - desired_m)) > 0:
                #     print("error")

            elif select_flag[i] == False:
                seq_local = func_seq_local(seq_population_g[i, :], mask, Q)
                K = seq_local.shape[1]
                rms_best = 99999
                for k in range(K):
                    seq_trial = seq_local[:,k]
                    rms_trial = func_objective(data['T'], seq_trial,Q, data['mask'], data['desired_amount'])
                    if rms_trial < rms_best:
                        seq_best = seq_trial
                        rms_best = rms_trial
                seq_population[i] = seq_best

                # tmp = np.minimum(mask[:, np.round(seq_best)],
                #                      np.ones_like(mask[:, np.round(seq_best)]))
                # num = np.sum(tmp, 1)
                # if np.sum(np.abs(num - desired_m)) > 0:
                #     print("error")
            #seq_best = seq_cur
        #for i in range(num_population):
            #product = seq_population[i, :]
            # tmp = mask[:, np.round(product)]
            # num = np.sum(tmp, 1)
            # if np.sum(np.abs(num - 1)) > 0:
            #     print("error")


    rms = np.zeros((num_population))
    for i in range(num_population):
        seq_cur = seq_population[i,:]
        rms[i] = func_objective(data['T'], seq_cur, Q, data['mask'], data['desired_amount'])
    print(np.min(rms),np.argmin(rms))
    return seq_population[np.argmin(rms),:], bestrms

if __name__ == '__main__':

    prob = 0.8
    num_generation = 15

    data = {}
    num_position = 50
    num_product_type = 8


     #np.arange(0, num_box, 1, dtype=int)#np.random.randint(0, num_product_type, num_box)
    data['desired_amount'] = np.random.randint(1, 4, (num_product_type))

    r = np.random.random((num_product_type, num_position))  # data.mask
    data['mask'] = np.zeros((num_product_type, num_position), dtype=int)
    while np.min(np.sum(data['mask'], 1) - data['desired_amount']) < 0:
        r = np.random.random((num_product_type, num_position))  # data.mask
        data['mask'] = np.zeros((num_product_type, num_position), dtype=int)
        for i in range(num_position):
            j = np.argmax(r[:, i])
            data['mask'][j, i] = np.random.randint(1, 5, (1))
        print(np.min(np.sum(data['mask'], 1)))

    data['num_box'] = np.sum(data['desired_amount'])

    A, B, C = 10, 50, 2
    Q = np.random.rand(num_position+1, 3)
    AllQ = np.random.rand(A*B*C, 3)
    x = np.linspace(0, A-1, 10)
    y = np.linspace(0, B-1, 50)
    vx, vy = np.meshgrid(x, y)
    for i in range(C):
        AllQ[i*A*B:(i+1)*A*B,0] = np.reshape(vx, (A*B))
        AllQ[i * A * B:(i + 1) * A * B, 1] = np.reshape(vy, (A*B))
        AllQ[i * A * B:(i + 1) * A * B, 2] = i
    randidx = np.random.permutation(A * B * C)
    Q[1:num_position+1] = AllQ[randidx[0:num_position],:]
    # Q[:, 0] = np.round(Q[:, 0] * (A-1))
    # Q[:, 1] = np.round(Q[:, 1] * (B-1))
    # Q[:, 2] = np.round(Q[:, 2] * (C-1))
    Q[0, :] = [-1, -1, 0]

    data['Q'] = Q
    ########### calculate the cost time between each position pair
    t1 = 2
    t2 = 10
    t3 = 0.5
    data['T'] = np.zeros((num_position+1, num_position+1, 2))
    for i in range(num_position+1):
        for j in range(num_position+1):
            data['T'][i, j, 0] = costtime_between_points(Q[i, :],  Q[j, :], 0, t1, t2, t3, B)  #turning time 20s
            data['T'][i, j, 1] = costtime_between_points(Q[i, :], Q[j, :], 1, t1, t2, t3, B)  #turning time 20s

    seq, bestrms = func_diff_evolution(data, prob, num_generation)

    print(data['T'])
    print(data['mask'])
    print(Q)
    print(seq)

    sio.savemat('save.mat',
                {'T': data['T'], 'desired_amount': data['desired_amount'], 'num_box': data['num_box'],
                 'mask': data['mask'], 'Q': Q, 'arrayY': seq, 'bestrms': bestrms})