import random
import math
import numpy as np
import sys
import copy
from config import *
import os


class MECenv(object):
    def __init__(self):

        self.adjust_count = 0
        self.current_vnf_i = -1
        self.current_k = -1
        self.pre_M = None
        self.A = None
        self.state = None
        self.K = 4
        self.C = np.array([6619, 4456, 4102, 7037])
        self.M = np.array([20, 17, 20, 19]) * 1024

        self.vnf_type_0 = 15
        self.M_base = np.array([5, 7, 4, 4, 4, 4, 5, 6, 5, 4, 6, 6, 5, 4, 8]) * 1024
        self.start_SFCs = np.load(os.path.join('config', 'SFCs_20.npy'))

        self.SFCs = np.vstack(self.start_SFCs).reshape(-1, 3)
        self.vnf_carry = np.array([0 for _ in range(len(self.SFCs))])
        self.choice_server = np.array([-1 for _ in range(len(self.SFCs))])
        self.current_M = copy.deepcopy(self.M)
        self.current_C = copy.deepcopy(self.C)
        self.pre_min_M = copy.deepcopy(self.M)
        self.current_w = np.array([0 for _ in range(self.K)])
        self.vnf_type = np.array([[-1, -1, -1, -1, -1, -1, -1, -1] for _ in range(self.K)])

        self.current_time = 0
        self.remain_time_vnf = np.array([0 for _ in range(len(self.start_SFCs))])

        self.start_vnf = np.array([-1.0 for _ in range(len(self.SFCs))])
        self.end_vnf = np.array([-1.0 for _ in range(len(self.SFCs))])
        self.delay_redeploy = 0.06
        self.delay_adjust = 0.01

        self.Q = [[] for _ in range(self.K)]

        self.current_C_busy = self.C - self.current_C
        self.current_w_busy = np.array([0 for _ in range(self.K)])

    def reset(self):
        self.adjust_count = 0
        self.SFCs = np.vstack(self.start_SFCs).reshape(-1, 3)
        self.start_vnf = np.array([-1.0 for _ in range(len(self.SFCs))])
        self.end_vnf = np.array([-1.0 for _ in range(len(self.SFCs))])
        self.Q = [[] for _ in range(self.K)]
        self.A_done = np.array([-1 for _ in range(len(self.SFCs))])
        self.exe_VNF = [[] for _ in range(len(self.SFCs))]

        self.SFCs = np.vstack(self.start_SFCs).reshape(-1, 3)
        self.vnf_carry = np.array([0 for _ in range(len(self.SFCs))])
        self.choice_server = np.array([-1 for _ in range(len(self.SFCs))])
        self.current_M = copy.deepcopy(self.M)
        self.current_C = copy.deepcopy(self.C)
        self.pre_min_M = copy.deepcopy(self.M)
        self.current_w = np.array([0 for _ in range(self.K)])
        self.vnf_type = np.array([[-1, -1, -1, -1, -1, -1, -1, -1] for _ in range(self.K)])
        self.pre_M = np.array([
            [self.M[i], self.M[i], self.M[i], self.M[i], self.M[i], self.M[i], self.M[i]] for i in
            range(self.K)])
        self.current_time = 0
        self.remain_time_vnf = np.array([0 for _ in range(len(self.start_SFCs))])

        SFCs_flat = self.SFCs.flatten().reshape(1, -1)

        # self.current_C_busy = self.C - self.current_C
        # self.current_w_busy = np.array([0 for _ in range(self.K)])

        self.state = np.concatenate([np.array(SFCs_flat),
                                     (self.M_base / 1024).reshape(1, -1),
                                     self.vnf_carry.reshape(1, -1),
                                     (self.current_M / 1024).reshape(1, -1),
                                     (self.current_C / 100.0).reshape(1, -1),
                                     self.current_w.reshape(1, -1),
                                     self.vnf_type.reshape(1, -1),
                                     (self.pre_min_M / 1024.0).reshape(1, -1),
                                     self.remain_time_vnf.reshape(1, -1)
                                     ],
                                    axis=1
                                    )

        return self.state

    def step(self, action):
        self.current_vnf_i, self.current_k = self.current_choice_vnf_server(action)
        reward = self.delay_t(action)

        finish = self.finish()
        if not finish:
            next_valid_actions = self.valid_A_t()
            while 1 not in next_valid_actions:
                self.next_time()
                next_valid_actions = self.valid_A_t()

        SFCs_flat = self.SFCs.flatten().reshape(1, -1)

        self.state = np.concatenate([np.array(SFCs_flat),
                                     (self.M_base / 1024).reshape(1, -1),
                                     self.vnf_carry.reshape(1, -1),
                                     (self.current_M / 1024).reshape(1, -1),
                                     (self.current_C / 100.0).reshape(1, -1),
                                     self.current_w.reshape(1, -1),
                                     self.vnf_type.reshape(1, -1),
                                     (self.pre_min_M / 1024.0).reshape(1, -1),
                                     self.remain_time_vnf.reshape(1, -1)
                                     ],
                                    axis=1
                                    )
        finish = self.finish()
        return self.state, -10 * reward, finish, 0

    def finish(self):
        for i in range(len(self.SFCs)):
            if self.SFCs[i][0] != 0 and self.vnf_carry[i] == 0:
                return False
        return True

    def delay_t(self, action):
        n, z = [], []
        vnf_i, k = self.current_vnf_i, self.current_k
        for j in range(len(self.vnf_carry)):
            if self.vnf_carry[j] == 1 and int(self.start_vnf[j] * 100) == self.current_time and self.choice_server[
                j] == k:
                n.append(j)
        n.append(vnf_i)
        for j in range(len(self.vnf_carry)):
            if self.vnf_carry[j] == 1:
                if int(self.start_vnf[j] * 100) < self.current_time < int(self.end_vnf[j] * 100):
                    if self.choice_server[j] == k:
                        z.append(j)

        n1, z1, adjust_n, adjust_z, relay_t = self.allocation_C(n, z, action)
        if len(z1) != 0:
            self.adjust_count += 1

        self.update_state(n1, z1, adjust_n, adjust_z, action)
        return relay_t

    def finishtime_vnf(self, i):
        exe_i = self.exe_VNF[i]
        w = self.SFCs[i][0]
        if len(exe_i) == 1:
            self.end_vnf[i] = int(exe_i[0][0] * 100) / 100.0 + w / exe_i[0][1]

            self.end_vnf[i] = math.ceil(round(self.end_vnf[i] * 100, 4)) / 100.0
            return self.end_vnf[i]
        else:
            w -= (exe_i[1][0] - exe_i[0][0]) * exe_i[0][1]
            for j in range(2, len(exe_i)):
                w -= (exe_i[j][0] - exe_i[j - 1][0] - self.delay_adjust) * exe_i[j - 1][1]
            self.end_vnf[i] = exe_i[-1][0] + self.delay_adjust + w / exe_i[-1][1]
            self.end_vnf[i] = math.ceil(round(self.end_vnf[i] * 100, 4)) / 100.0
            return self.end_vnf[i]

    def update_state(self, n, z_change, adjust_n, adjust_z, action):
        vnf_i, k = self.current_vnf_i, self.current_k
        for i in range(len(n)):
            if n[i] not in self.Q[k]:
                self.Q[k].append(n[i])
                self.start_vnf[n[i]] = self.current_time / 100.0
                self.exe_VNF[n[i]].append([self.current_time / 100.0, adjust_n[i]])
            else:
                self.exe_VNF[n[i]][-1][1] = adjust_n[i]
            self.end_vnf[n[i]] = self.finishtime_vnf(n[i])
            self.remain_time_vnf[n[i] // 6] = int(self.end_vnf[n[i]] * 100) - self.current_time

        for i in range(len(z_change)):
            if self.exe_VNF[z_change[i]][-1][0] == self.current_time / 100.0:
                self.exe_VNF[z_change[i]][-1][1] = adjust_z[i]
            else:
                self.exe_VNF[z_change[i]].append([self.current_time / 100.0, adjust_z[i]])
            self.end_vnf[z_change[i]] = self.finishtime_vnf(z_change[i])
            self.remain_time_vnf[z_change[i] // 6] = int(self.end_vnf[z_change[i]] * 100) - self.current_time

        self.vnf_carry[vnf_i] = 1
        self.choice_server[vnf_i] = k
        self.current_M[k] -= self.SFCs[vnf_i][1] + self.M_base[self.SFCs[vnf_i][2]]
        self.pre_min_M[k] = min(self.pre_min_M[k], self.current_M[k])
        self.current_C[k] = np.sum(adjust_n)
        self.current_w[k] += self.SFCs[vnf_i][0]
        if self.SFCs[vnf_i][2] in self.vnf_type[k]:
            for j in range(len(self.vnf_type[k])):
                if self.vnf_type[k][j] == self.SFCs[vnf_i][2]:
                    self.vnf_type[k][j] = -1
                    if j > 0 and self.vnf_type[k][j - 1] != -1:
                        for p in range(j, 0, -1):
                            self.vnf_type[k][p] = self.vnf_type[k][p - 1]
                        self.vnf_type[k][0] = -1

                    break
        else:
            pass

        sum = 0
        for j in range(len(self.vnf_type[k])):
            if self.vnf_type[k][j] != -1:
                sum += self.M_base[self.vnf_type[k][j]]
                if sum > self.current_M[k]:
                    self.vnf_type[k][j] = -1
        while self.vnf_type[k][len(self.vnf_type[k]) - 1] == -1:
            p = -1
            for j in range(len(self.vnf_type[k])):
                if self.vnf_type[k][j] != -1:
                    p = j
            if p == -1:
                break
            for j in range(len(self.vnf_type[k]) - 1, 0, -1):
                self.vnf_type[k][j] = self.vnf_type[k][j - 1]
            self.vnf_type[k][0] = -1

    def next_time(self):
        self.current_w = np.array([0 for _ in range(self.K)])
        for i in range(len(self.start_SFCs)):
            if self.remain_time_vnf[i] != 0:
                self.remain_time_vnf[i] -= 1

        for i in range(self.K):
            self.pre_M[i][len(self.pre_M[i]) - 1] = self.current_M[i]

        self.current_M = copy.deepcopy(self.M)
        self.current_C = copy.deepcopy(self.C)
        for i in range(len(self.vnf_carry)):
            if int(self.end_vnf[i] * 100) > (self.current_time + 1):
                k = self.choice_server[i]
                self.current_M[k] -= self.SFCs[i][1] + self.M_base[self.SFCs[i][2]]
                self.current_C[k] -= self.exe_VNF[i][-1][1]

        min_M = np.array([0 for i in range(self.K)])
        for i in range(self.K):
            for j in range(len(self.pre_M[i]) - 1):
                self.pre_M[i][j] = self.pre_M[i][j + 1]
            self.pre_M[i][len(self.pre_M[i]) - 1] = self.current_M[i]
            min_M[i] = np.min(self.pre_M[i])
            self.pre_min_M[i] = min_M[i]

        self.vnf_type = copy.deepcopy(self.vnf_type)

        for i in range(len(self.vnf_carry)):
            if int(self.end_vnf[i] * 100) == (self.current_time + 1):
                k = self.choice_server[i]
                j = np.where(self.vnf_type[k] == -1)[0][-1]
                self.vnf_type[k][j] = self.SFCs[i][2]

        self.current_time += 1

    def tuyouhua(self, n, z, action):
        vnf_i, k = self.current_vnf_i, self.current_k
        tem_c = 0
        for i in range(len(z)):
            if self.current_time - int(self.exe_VNF[z[i]][-1][0] * 100) == 0:
                tem_c += self.exe_VNF[z[i]][-2][1]
            else:
                tem_c += self.exe_VNF[z[i]][-1][1]

        n_vnf, z_vnf = [], []
        for i in range(len(n)):
            n_vnf.append(self.SFCs[n[i]])
        for i in range(len(z)):
            z_vnf.append(self.SFCs[z[i]])
        n_vnf = np.array(n_vnf)
        z_vnf = np.array(z_vnf)

        t_z_pro = []
        w_z = []
        for i in range(len(z)):
            exe_i = self.exe_VNF[z[i]]

            w_i = self.SFCs[z[i]][0]
            if len(exe_i) == 1:
                w_i -= (self.current_time / 100.0 - exe_i[0][0]) * exe_i[0][1]
                t_z_pro.append(self.end_vnf[z[i]])
                w_z.append(w_i)
            else:
                w_i -= (exe_i[1][0] - exe_i[0][0]) * exe_i[0][1]
                for j in range(2, len(exe_i)):
                    w_i -= (exe_i[j][0] - exe_i[j - 1][0] - self.delay_adjust) * exe_i[j - 1][1]
                if exe_i[-1][0] < self.current_time / 100.0:
                    w_i -= (self.current_time / 100.0 - exe_i[-1][0] - self.delay_adjust) * exe_i[-1][1]
                else:
                    w_i = w_i
                t_z_pro.append(self.end_vnf[z[i]])
                w_z.append(w_i)

        w_z = np.array(w_z)
        w_sum = 0.
        for i in range(len(n_vnf)):
            w_sum += n_vnf[i][0] ** 0.5
        for i in range(len(w_z)):
            w_sum += w_z[i] ** 0.5
        adjust_n = ((n_vnf[:, 0] ** 0.5) * (tem_c + self.current_C[k])) / w_sum
        adjust_z = ((w_z ** 0.5) * (tem_c + self.current_C[k])) / w_sum
        adjust_n_floor = np.floor(adjust_n)
        adjust_z_floor = np.floor(adjust_z)

        sum_c = sum(adjust_n_floor) + sum(adjust_z_floor)
        res = int(tem_c + self.current_C[k] - sum_c)
        j = 0
        for i in range(len(adjust_z_floor)):
            if adjust_z_floor[i] == 0 and res > 0:
                adjust_z_floor[i] = 1
                res -= 1
            if adjust_z_floor[i] == 0 and res == 0:
                adjust_z_floor[i] = 1
                while adjust_n_floor[j % len(adjust_n_floor)] <= 1:
                    j += 1
                adjust_n_floor[j % len(adjust_n_floor)] -= 1
                j += 1
        if res > 0:
            for j in range(res):
                adjust_n_floor[j % len(adjust_n_floor)] += 1

        t_n = np.sum(n_vnf[:, 0] / adjust_n_floor + self.current_time / 100.0)
        t_z = self.current_time / 100.0 + self.delay_adjust + w_z / adjust_z_floor
        t_z_change = np.sum(t_z - t_z_pro)

        t_sum = 0.
        for j in range(len(n)):
            if n[j] != self.current_vnf_i:
                t_sum += self.end_vnf[n[j]]

        tt = t_n - t_sum
        if self.current_vnf_i % 6 != 0:
            tt = t_n - t_sum - self.end_vnf[self.current_vnf_i - 1]

        return adjust_n_floor, adjust_z_floor, tt + t_z_change

    def allocation_C(self, n, z, action):
        adjust_n, adjust_z, t_0 = self.tuyouhua(n, [], action)
        if len(z) == 0:
            return n, z, adjust_n, adjust_z, t_0

        tt_change = []
        for i in range(len(z)):
            tem_z = [z[i]]
            adjust_n, adjust_z, t_1 = self.tuyouhua(n, tem_z, action)
            tt = t_0 - t_1
            tt_change.append([tt, z[i]])

        tt_change = np.array(tt_change)
        tt_change = np.array(tt_change)
        tt_change = tt_change[tt_change[:, 0].argsort()[::-1]]

        if tt_change[0][0] <= 0:
            adjust_n, adjust_z, t_0 = self.tuyouhua(n, [], action)
            return n, [], adjust_n, adjust_z, t_0

        z_change = []
        pre_t = tt_change[0][0]
        for i in range(len(tt_change)):

            if tt_change[i][0] > 0:
                z_change.append(int(tt_change[i][1]))
            else:
                break

            if i == 0:
                continue
            adjust_n, adjust_z, t_2 = self.tuyouhua(n, z_change, action)
            if t_0 - t_2 > pre_t:
                pre_t = t_0 - t_2
                continue
            else:
                z_change.pop()
                adjust_n, adjust_z, t_2 = self.tuyouhua(n, z_change, action)
                return n, z_change, adjust_n, adjust_z, t_2

        adjust_n, adjust_z, t = self.tuyouhua(n, z_change, action)
        return n, z_change, adjust_n, adjust_z, t

    def current_choice_vnf_server(self, action):
        i, k = action // self.K, action % self.K
        for j in range(6):
            if self.vnf_carry[i * 6 + j] == 0 and self.SFCs[i * 6 + j][0] != 0:
                return i * 6 + j, k

    def valid_A_t(self):
        self.A = np.array([int(0) for _ in range(self.K * len(self.start_SFCs))])
        for i in range(len(self.start_SFCs)):
            if self.remain_time_vnf[i] == 0:
                for j in range(6):
                    if self.vnf_carry[i * 6 + j] == 0 and self.SFCs[i * 6 + j][0] != 0:
                        vnf = self.SFCs[i * 6 + j]
                        for k in range(self.K):
                            if self.current_w[k] != 0 and self.current_C[k] < 100:
                                continue
                            if vnf[2] in self.vnf_type[k] or self.pre_min_M[k] >= vnf[1] + self.M_base[
                                vnf[2]]:
                                if self.current_C[k] > 0 and self.current_M[k] >= vnf[1] + self.M_base[
                                    vnf[2]]:
                                    self.A[i * self.K + k] = 1
                        break
        return self.A

    def computing_sum_relay(self):
        sum = 0
        for i in range(len(self.start_SFCs)):
            for j in range(6):
                if j == 5 and self.SFCs[i * 6 + j][0] != 0:
                    sum += self.end_vnf[i * 6 + j]
                    break
                if self.SFCs[i * 6 + j][0] != 0 and self.SFCs[i * 6 + j + 1][0] == 0:
                    sum += self.end_vnf[i * 6 + j]
                    break
        return sum

# env = MECenv()
# state = env.reset()
# print(env.SFCs)
# print(state)
# print(state.shape[1])
# print(len(env.SFCs) * 4)
