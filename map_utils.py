# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 16:17:20 2022

@author: Thinkpad
"""

import numpy as np
import pandas as pd
import netCDF4
import time


class LagrangianMap:
    def __init__(self, nc_path, lat_step=0.08, lon_step=0.08, N_row=250, N_col=500, weight_factor=3.62, start_day=0):
        file2read = netCDF4.Dataset(nc_path, 'r')
        N_id = len(file2read.variables['lat'])
        # print(file2read.variables['lat'].shape, file2read.variables['lon'].shape, file2read.variables['p_id'].shape)
        self.Lat_matrix = file2read.variables['lat']
        self.Lon_matrix = file2read.variables['lon']
        self.P_id_list = file2read.variables['p_id']
        self.file2read = file2read
        self.lat_step, self.lon_step = lat_step, lon_step
        self.N_row, self.N_col = N_row, N_col
        self.N_id = len(file2read.variables['lat'])
        self.weight_factor = weight_factor
        self.sea_current_model = 'NA'
        self.wind_coef = 'NA'
        ### Since the mapL data is truncated, we need to specify the start day.
        self.start_day = start_day

    def path_adj_reward_alpha(self, node_i, node_j, i_start, j_start, T0, Height_t, alpha, remain_index0=-1):
        """
        calculate reward for given path, eg in top(12) algo
        Height_t should be the small truncated map

        Note (2023 Aug 1): This function was not supporting different initial remain_idx, namely,
        it by default assume that there is no collection before. We fix this, but need to check all functions related to it.
        """
        if isinstance(remain_index0, int) and remain_index0 == -1:
            remain_index0 = np.ones(self.N_id).astype('bool')
        remain_index0 = np.ma.array(data=remain_index0, fill_value=0)

        T_plan = len(node_i)
        speed_scalar = 1
        assert 0 <= alpha <= 1
        lat_step_sc, lon_step_sc = self.lat_step / speed_scalar, self.lon_step / speed_scalar
        R_seq_raw = -np.ones(T_plan)
        R_seq_adj = -np.ones(T_plan)
        # remain_idx = np.ones(self.N_id) this is the old code we use before Aug 1 2023
        remain_idx = remain_index0.copy()

        for t in range(T_plan):
            # Location of all particle
            day, step = (T0 + t) // 8, (T0 + t) % 8
            Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8 - step) + step * self.Lat_matrix[:, day-self.start_day + 1]) / 8
            Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8 - step) + step * self.Lon_matrix[:, day-self.start_day + 1]) / 8

            # Location of visit cell:
            i_loc, j_loc = node_i[t] // speed_scalar, node_j[t] // speed_scalar
            i_real, j_real = i_start + i_loc, j_start + j_loc

            lat_l, lat_u = 39.92 - self.lat_step * i_real - self.lat_step / 2, 39.92 - self.lat_step * i_real + self.lat_step / 2
            lon_l, lon_u = -160 + self.lon_step * j_real - self.lon_step / 2, -160 + self.lon_step * j_real + self.lon_step / 2

            # 6m
            wave_height = Height_t[i_real, j_real, T0+t-8*self.start_day]# Height_t[node_i[t], node_j[t], T0+t]Height_t[node_i[t], node_j[t], T0 + t]
            if wave_height <= 6:
                # Which particles in current cell
                current_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u) & (Lon_intra > lon_l) & (Lon_intra <= lon_u)
                R_seq_raw[t] = np.sum(current_idx)

                # Which remain particles in current cell
                current_idx_adj = remain_idx * current_idx
                R_seq_adj[t] = alpha * np.sum(current_idx_adj)

                remain_idx = remain_idx - alpha * current_idx_adj

            else:
                R_seq_raw[t], R_seq_adj[t] = 0, 0

        return R_seq_raw, R_seq_adj

    def path_adj_reward_alpha_test(self, node_i, node_j, i_start, j_start, T0, Height_t, alpha, remain_index0=-1):
        """
        calculate reward for given path, eg in top(12) algo
        Height_t should be the small truncated map

        Note (2023 Aug 1): This function was not supporting different initial remain_idx, namely,
        it by default assume that there is no collection before. We fix this, but need to check all functions related to it.
        """
        time_0,time_1,time_2,time_3,time_4=0,0,0,0,0 
        time_start = time.perf_counter()
        if isinstance(remain_index0, int) and remain_index0 == -1:
            remain_index0 = np.ones(self.N_id).astype('bool')
        remain_index0 = np.ma.array(data=remain_index0, fill_value=0)
        time_0 = time.perf_counter() - time_start

        T_plan = len(node_i)
        speed_scalar = 1
        assert 0 <= alpha <= 1
        lat_step_sc, lon_step_sc = self.lat_step / speed_scalar, self.lon_step / speed_scalar
        R_seq_raw = -np.ones(T_plan)
        R_seq_adj = -np.ones(T_plan)
        # remain_idx = np.ones(self.N_id) this is the old code we use before Aug 1 2023
        remain_idx = remain_index0.copy()
        
        for t in range(T_plan):
            # Location of all particle
            day, step = (T0 + t) // 8, (T0 + t) % 8
            Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8 - step) + step * self.Lat_matrix[:, day-self.start_day + 1]) / 8
            Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8 - step) + step * self.Lon_matrix[:, day-self.start_day + 1]) / 8

            # Location of visit cell:
            i_loc, j_loc = node_i[t] // speed_scalar, node_j[t] // speed_scalar
            i_real, j_real = i_start + i_loc, j_start + j_loc

            lat_l, lat_u = 39.92 - self.lat_step * i_real - self.lat_step / 2, 39.92 - self.lat_step * i_real + self.lat_step / 2
            lon_l, lon_u = -160 + self.lon_step * j_real - self.lon_step / 2, -160 + self.lon_step * j_real + self.lon_step / 2

            # 6m
            wave_height = Height_t[i_real, j_real, T0+t-8*self.start_day]# Height_t[node_i[t], node_j[t], T0+t]Height_t[node_i[t], node_j[t], T0 + t]
            if wave_height <= 6:
                # Which particles in current cell
                time_start = time.perf_counter()
                current_idx = np.full(Lat_intra.shape, False)
                sub_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u)
                current_idx[sub_idx] =  (Lon_intra[sub_idx] > lon_l) & (Lon_intra[sub_idx] <= lon_u)
    
                # current_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u) & (Lon_intra > lon_l) & (Lon_intra <= lon_u)
                time_1 += time.perf_counter() - time_start

                time_start = time.perf_counter()
                R_seq_raw[t] = np.sum(current_idx)
                time_2 += time.perf_counter() - time_start

                time_start = time.perf_counter()
                # Which remain particles in current cell
                current_idx_adj = remain_idx * current_idx
                time_3 += time.perf_counter() - time_start

                R_seq_adj[t] = alpha * np.sum(current_idx_adj)

                time_start = time.perf_counter()
                remain_idx = remain_idx - alpha * current_idx_adj
                time_4 += time.perf_counter() - time_start

            else:
                R_seq_raw[t], R_seq_adj[t] = 0, 0

        # print(time_0,time_1,time_2,time_3,time_4)
        return R_seq_raw, R_seq_adj

        

    def path_adj_reward_ept_alpha(self, node_i, node_j, i_start, j_start, T0, a_list, current_load0, total_load, Height_t, alpha, remain_index0=-1):
        """
        Here we consider different action
        The plastic will be removed only when action==0

        Note:   When caculate the load in the retention zone, we do consider the weight factor.
                When caculate the reward vec, we do not times the weight factor. Need to times it outside.
        """
        if isinstance(remain_index0, int) and remain_index0 == -1:
            remain_index0 = np.ones(self.N_id).astype('bool')
        remain_index0 = np.ma.array(data=remain_index0, fill_value=0)

        T_plan = len(node_i)
        speed_scalar = 1
        # lat_step_sc, lon_step_sc = self.lat_step / speed_scalar, self.lon_step / speed_scalar
        R_seq_raw = -np.ones(T_plan)
        R_seq_adj = -np.ones(T_plan)
        # remain_idx = self.Lat_matrix[:, 0] < 1000 this is the old code we use before Aug 1 2023
        remain_idx = remain_index0.copy()
        current_load = current_load0

        for t in range(T_plan):

            if a_list[t] in [0, 1, 3]: # no matter which, it may be able to collect:
            # Location of all particle
                day, step = (T0 + t) // 8, (T0 + t) % 8
                Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8 - step) + step * self.Lat_matrix[:, day-self.start_day + 1]) / 8
                Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8 - step) + step * self.Lon_matrix[:, day-self.start_day + 1]) / 8

                # Location of visit cell:
                i_loc, j_loc = node_i[t] // speed_scalar, node_j[t] // speed_scalar
                i_real, j_real = i_start + i_loc, j_start + j_loc

                lat_l, lat_u = 39.92 - self.lat_step * i_real - self.lat_step / 2, 39.92 - self.lat_step * i_real + self.lat_step / 2
                lon_l, lon_u = -160 + self.lon_step * j_real - self.lon_step / 2, -160 + self.lon_step * j_real + self.lon_step / 2

                # Which particles in current cell
                current_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u) & (Lon_intra > lon_l) & (Lon_intra <= lon_u)
                R_seq_raw[t] = np.sum(current_idx)

                wave_height = Height_t[i_real, j_real, T0+t-8*self.start_day] # Height_t[node_i[t], node_j[t], T0+t]Height_t[node_i[t], node_j[t], T0+t]
                if wave_height <= 6:

                    # Which remain particles in current cell
                    current_idx_adj = remain_idx * current_idx
                    adj_reward = alpha * np.sum(current_idx_adj)
                    if current_load + adj_reward * self.weight_factor <= total_load:
                        R_seq_adj[t] = adj_reward
                        remain_idx = remain_idx - alpha * current_idx_adj

                        current_load += adj_reward * self.weight_factor
                    else:
                        R_seq_adj[t] = 0
                else:
                    # Wave > 6m, no collection for sure
                    R_seq_raw[t], R_seq_adj[t] = 0, 0

            else:
                if a_list[t] == 2:
                    current_load = 0
                R_seq_adj[t] = 0
                R_seq_raw[t] = 0

        return R_seq_raw, R_seq_adj, 0


    def generate_sub_map(self, i0, j0, T0, T_plan, remain_index0=-1):
        """
        Given a starting point (i0, j0, T0), generate the map in future T_plan time steps.
        remain_index0: the boolen vector indicating which particles are still stay in the ocean.

        Here we only consider the bool index, namely, the particles are either in the ocean or not.
        If the particle is partially removed, please use generate_sub_map_alpha.
        """
        if isinstance(remain_index0, int) and remain_index0 == -1:
            remain_index0 = np.ones(self.N_id).astype(bool)
        D_t_plan = np.zeros((self.N_row, self.N_col, T_plan))
        speed_scalar = 1
        i_start, i_end = int(max((0, i0 - T_plan/speed_scalar))), int(min((self.N_row, i0 + T_plan/speed_scalar)))
        j_start, j_end = int(max((0, j0 - T_plan/speed_scalar))), int(min((self.N_col, j0 + T_plan/speed_scalar)))
        i0_adj, j0_adj = speed_scalar*(i0-i_start), speed_scalar*(j0-j_start)

    #     print(i0,j0,T0,i_start, i_end, j_start, j_end)
        # Be careful... Here we ignore fist row and last col from orginal map.
        # In theory lat_start - lat_end = 2*lat_step/2 +
        lat_start, lat_end = 39.96 - i_start*self.lat_step, 39.96 - i_end*self.lat_step
        lon_start, lon_end = -160.04 + j_start*self.lon_step, -160.04 + j_end*self.lon_step

        for t in range(T_plan):
            day, step = (T0+t)//8, (T0+t)%8

            Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8-step) + step*self.Lat_matrix[:, day-self.start_day+1])/8
            Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8-step) + step*self.Lon_matrix[:, day-self.start_day+1])/8

    #         subset_idx = (Lon_intra <=-120.04) & (Lon_intra >=-160.04) & (Lat_intra <= 39.96) & (Lat_intra >= 19.96)
            subset_idx = (Lon_intra <=lon_end) & (Lon_intra >=lon_start) & (Lat_intra <= lat_start) & (Lat_intra >= lat_end)

    #         print(np.sum(subset_idx), np.sum(remain_index & subset_idx),len(subset_idx), len(remain_index & subset_idx))
            subset_idx = remain_index0 & subset_idx

            lat_valid = Lat_intra[subset_idx].data
            lon_valid = Lon_intra[subset_idx].data
    #         print(lon_end,lon_valid[:5])
            for lat_id, lon_id in zip(lat_valid, lon_valid):
                ii, jj = int((39.96-lat_id)//self.lat_step), int((lon_id + 160.04)//self.lon_step)
                # if ii<i_start or ii>=i_end or jj<j_start or jj>=j_end:
                #     print(ii, jj, lat_id, lon_id, i_start, i_end,  j_start, j_end)
                # try:
                D_t_plan[ii,jj,t] += 1

        return D_t_plan[i_start:i_end, j_start:j_end,:], i_start, j_start, i0_adj, j0_adj ,i_end, j_end

    def generate_sub_map_alpha(self, i0, j0, T0, T_plan, remain_index0=-1):
        """
        Given a starting point (i0, j0, T0), generate the map in future T_plan time steps.
        remain_index0: the boolen vector indicating which particles are still stay in the ocean.

        Here the index is between [0,1], indicating the fraction of particles in the ocean.
        To use bool index, please use generate_sub_map.
        """
        if isinstance(remain_index0, int) and remain_index0 == -1:
            remain_index0 = np.ones(self.N_id).astype('bool')
        remain_index0 = np.ma.array(data = remain_index0, fill_value=0)
        np.ma.set_fill_value(remain_index0, 0)
        D_t_plan = np.zeros((self.N_row, self.N_col, T_plan))
        speed_scalar = 1
        i_start, i_end = int(max((0, i0 - T_plan/speed_scalar))), int(min((self.N_row, i0 + T_plan/speed_scalar)))
        j_start, j_end = int(max((0, j0 - T_plan/speed_scalar))), int(min((self.N_col, j0 + T_plan/speed_scalar)))
        i0_adj, j0_adj = speed_scalar*(i0-i_start), speed_scalar*(j0-j_start)

    #     print(i0,j0,T0,i_start, i_end, j_start, j_end)
        # Be careful... Here we ignore fist row and last col from orginal map.
        # In theory lat_start - lat_end = 2*lat_step/2 +
        lat_start, lat_end = 39.96 - i_start*self.lat_step, 39.96 - i_end*self.lat_step
        lon_start, lon_end = -160.04 + j_start*self.lon_step, -160.04 + j_end*self.lon_step

        for t in range(T_plan):
            day, step = (T0+t)//8, (T0+t)%8

            Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8-step) + step*self.Lat_matrix[:, day-self.start_day+1])/8
            Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8-step) + step*self.Lon_matrix[:, day-self.start_day+1])/8

    #         subset_idx = (Lon_intra <=-120.04) & (Lon_intra >=-160.04) & (Lat_intra <= 39.96) & (Lat_intra >= 19.96)
            subset_idx0 = (Lon_intra <=lon_end) & (Lon_intra >=lon_start) & (Lat_intra <= lat_start) & (Lat_intra >= lat_end)
            np.ma.set_fill_value(subset_idx0, 0)
    #         print(np.sum(subset_idx), np.sum(remain_index & subset_idx),len(subset_idx), len(remain_index & subset_idx))
            subset_weight = (remain_index0.filled()) * (subset_idx0.filled())
            subset_idx = subset_weight > 0
            lat_valid = Lat_intra[subset_idx].data
            lon_valid = Lon_intra[subset_idx].data
            weight_list = remain_index0[subset_idx]
            # print(lon_end,lon_valid[:5], Lon_intra[subset_idx].data[:5], np.sum(subset_idx), np.sum(subset_weight), subset_idx[:5], subset_weight[:5])
            for lat_id, lon_id, weight_i in zip(lat_valid, lon_valid, weight_list):
                # if (np.isnan(lat_id) or np.isnan(lon_id)):
                #     return (subset_weight, remain_index0, subset_idx,subset_idx0 , Lat_intra)

                ii, jj = int((39.96-lat_id)//self.lat_step), int((lon_id + 160.04)//self.lon_step)
                # if ii<i_start or ii>=i_end or jj<j_start or jj>=j_end:
                #     print(ii, jj, lat_id, lon_id, i_start, i_end,  j_start, j_end)
                # try:
                D_t_plan[ii, jj, t] += weight_i

        return D_t_plan[i_start:i_end, j_start:j_end, :], i_start, j_start, i0_adj, j0_adj ,i_end, j_end


    def generate_static_sub_map_alpha(self, T0, i_start=0, i_end=-1, j_start=0, j_end=-1, remain_index0=-1):
        """
        Generate one static map with given time T0, with area defined by {i,j}_{start,end}
        More flexible than generate_sub_map_alpha, as you can customized the area;
        Currently used for normalization: get the particles in GPGP at the begining of the year.
        From 2022 Apr, we always use the alpha version, namely, the remain_index is fraction in [0,1], not {0,1}
        remain_index0: the float vector indicating particles' remaining weight (0-100%) in the ocean.
        """
        i_end, j_end = self.N_row if i_end==-1 else i_end, self.N_col if j_end==-1 else j_end
        if isinstance(remain_index0, int) and remain_index0 == -1:
            remain_index0 = np.ones(self.N_id).astype('bool')
        remain_index0 = np.ma.array(data = remain_index0, fill_value=0)
        np.ma.set_fill_value(remain_index0, 0)
        D_t_plan = np.zeros((self.N_row+1, self.N_col+1)) # We add one more cell to avoid rounding error.
        speed_scalar = 1

        # i0_adj, j0_adj = speed_scalar*(i0-i_start), speed_scalar*(j0-j_start)

    #     print(i0,j0,T0,i_start, i_end, j_start, j_end)
        # Be careful... Here we ignore fist row and last col from orginal map.
        # In theory lat_start - lat_end = 2*lat_step/2 +
        lat_start, lat_end = 39.96 - i_start*self.lat_step, 39.96 - i_end*self.lat_step
        lon_start, lon_end = -160.04 + j_start*self.lon_step, -160.04 + j_end*self.lon_step

        t=0
        day, step = (T0+t)//8, (T0+t)%8

        Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8-step) + step*self.Lat_matrix[:, day-self.start_day+1])/8
        Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8-step) + step*self.Lon_matrix[:, day-self.start_day+1])/8

#         subset_idx = (Lon_intra <=-120.04) & (Lon_intra >=-160.04) & (Lat_intra <= 39.96) & (Lat_intra >= 19.96)
        subset_idx0 = (Lon_intra <=lon_end) & (Lon_intra >=lon_start) & (Lat_intra <= lat_start) & (Lat_intra >= lat_end)
        np.ma.set_fill_value(subset_idx0, 0)
#         print(np.sum(subset_idx), np.sum(remain_index & subset_idx),len(subset_idx), len(remain_index & subset_idx))
        subset_weight = (remain_index0.filled()) * (subset_idx0.filled())
        subset_idx = subset_weight > 0
        lat_valid = Lat_intra[subset_idx].data
        lon_valid = Lon_intra[subset_idx].data
        weight_list = remain_index0[subset_idx]
        # print(lon_end,lon_valid[:5], Lon_intra[subset_idx].data[:5], np.sum(subset_idx), np.sum(subset_weight), subset_idx[:5], subset_weight[:5])
        for lat_id, lon_id, weight_i in zip(lat_valid, lon_valid, weight_list):
            # if (np.isnan(lat_id) or np.isnan(lon_id)):
            #     return (subset_weight, remain_index0, subset_idx,subset_idx0 , Lat_intra)

            ii, jj = int((39.96-lat_id)//self.lat_step), int((lon_id + 160.04)//self.lon_step)
            # if ii<i_start or ii>=i_end or jj<j_start or jj>=j_end:
            #     print(ii, jj, lat_id, lon_id, i_start, i_end,  j_start, j_end)
            # try:
            D_t_plan[ii, jj] += weight_i

        return D_t_plan[i_start:i_end, j_start:j_end]


    def update_remain_index_alpha(self, remain_index0, path_df, reward_df, T_begin_loc,
                                 Height_t, alpha):
        remain_idx = remain_index0
        lat_step, lon_step = 0.08, 0.08
        added_load = []
        current_load = 0
        IS_removed = False
        for index, row in path_df.iterrows():
            if pd.isna(row['t']):
                break
            elif row['t'] >= T_begin_loc:
                if row['action'] in [0, 1, 3]:
                    t = row['t']
                    day, step = t // 8, t % 8
                    try:
                        Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8 - step) + step * self.Lat_matrix[:, day-self.start_day + 1]) / 8
                        Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8 - step) + step * self.Lon_matrix[:, day-self.start_day + 1]) / 8
                    except:
                        print("Error!!! Cannot create Lat_intra: ", day, step, t)
                    # Location of visit cell:

                    i_real, j_real = row['node_i'], row["node_j"]

                    lat_l, lat_u = 39.92 - lat_step * i_real - lat_step / 2, 39.92 - lat_step * i_real + lat_step / 2
                    lon_l, lon_u = -160 + lon_step * j_real - lon_step / 2, -160 + lon_step * j_real + lon_step / 2
                    wave_height = Height_t[i_real, j_real, t-8*self.start_day]
                    if wave_height <= 6:
                        # Which particles in current cell
                        current_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u) & (Lon_intra > lon_l) & (Lon_intra <= lon_u)

                        reward_df.loc[index, "raw"] = np.sum(current_idx)

                        # Which remain particles in current cell
                        current_idx_adj = remain_idx * current_idx
                        adj_reward = alpha * np.sum(current_idx_adj)

                        reward_df.loc[index, "adj"] = adj_reward
                        # remain_idx = remain_idx & (1 - current_idx)  # remove plastic
                        remain_idx = remain_idx - alpha * current_idx_adj
                        # added_load.append(adj_reward )
                        current_load += adj_reward
                        IS_removed = True

                            # Full, not update index, not update load
                        reward_df.loc[index, "load"] = current_load
                    else:
                        # wave height > 6m
                        # current_idx = (Lat_intra > 1000) & (Lat_intra < -1000)
                        # remain_idx = remain_idx & (1 - current_idx)
                        reward_df.loc[index, "raw"], reward_df.loc[index, "adj"] = 0, 0
                        reward_df.loc[index, "load"] = current_load
                else:

                    reward_df.loc[index, "raw"] = 0
                    reward_df.loc[index, "adj"] = 0
                    reward_df.loc[index, "load"] = current_load

        speed_load = 0
        # if IS_removed:
        #     return remain_idx.data.astype(bool), reward_df.copy(), current_load, speed_load
        # else:
        return remain_idx, reward_df.copy()#, current_load, speed_load  # .data.astype(bool)


    def update_remain_index_ept_alpha(self, remain_index0, path_df, reward_df, T_begin_loc, current_load0,total_load,
                                speed_loadt, speed_load0, weight_factor, Height_t, alpha):
        remain_idx = remain_index0
        lat_step, lon_step = 0.08, 0.08
        added_load = []
        current_load = current_load0
        IS_removed = False
        for index, row in path_df.iterrows():
            if pd.isna(row['t']):
                break
            elif row['t'] >= T_begin_loc:
                if row['action'] in [0, 1, 3]:
                    t = row['t']
                    day, step = t // 8, t % 8
                    try:
                        Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8 - step) + step * self.Lat_matrix[:, day-self.start_day + 1]) / 8
                        Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8 - step) + step * self.Lon_matrix[:, day-self.start_day + 1]) / 8
                    except:
                        print("Error!!! Cannot create Lat_intra: ", day, step, t)
                    # Location of visit cell:

                    i_real, j_real = row['node_i'], row["node_j"]

                    lat_l, lat_u = 39.92 - lat_step * i_real - lat_step / 2, 39.92 - lat_step * i_real + lat_step / 2
                    lon_l, lon_u = -160 + lon_step * j_real - lon_step / 2, -160 + lon_step * j_real + lon_step / 2
                    wave_height = Height_t[i_real, j_real, t-8*self.start_day]
                    if wave_height <= 6:
                        # Which particles in current cell
                        current_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u) & (Lon_intra > lon_l) & (Lon_intra <= lon_u)
                        if row['action'] == 0:
                            reward_df.loc[index, "raw"] = np.sum(current_idx)
                        if row['action'] in [1, 3]:
                            reward_df.loc[index, "raw"] = 0
                        # Which remain particles in current cell
                        current_idx_adj = remain_idx * current_idx
                        adj_reward = alpha * np.sum(current_idx_adj)
                        if current_load + adj_reward * weight_factor <= total_load:  # not exceed
                            reward_df.loc[index, "adj"] = adj_reward
                            # remain_idx = remain_idx & (1 - current_idx)  # remove plastic
                            remain_idx = remain_idx - alpha * current_idx_adj
                            added_load.append(adj_reward * weight_factor)
                            current_load += adj_reward * weight_factor
                            IS_removed = True
                        else:
                            reward_df.loc[index, "adj"] = 0
                            # Full, not update index, not update load
                        reward_df.loc[index, "load"] = current_load
                    else:
                        # wave height > 6m
                        # current_idx = (Lat_intra > 1000) & (Lat_intra < -1000)
                        # remain_idx = remain_idx & (1 - current_idx)
                        reward_df.loc[index, "raw"], reward_df.loc[index, "adj"] = 0, 0
                        reward_df.loc[index, "load"] = current_load
                else:
                    if row['action'] == 2:
                        current_load = 0
                    reward_df.loc[index, "raw"] = 0
                    reward_df.loc[index, "adj"] = 0
                    reward_df.loc[index, "load"] = current_load

        speed_load = 8 / 3 * np.average(added_load) + (speed_load0 + speed_loadt) / 3 if np.sum(added_load) > 0 else (speed_load0 + speed_loadt) / 2
        if IS_removed:
            return remain_idx.data.astype(bool), reward_df.copy(), current_load, speed_load
        else:
            return remain_idx, reward_df.copy(), current_load, speed_load  # .data.astype(bool)


    def get_current_idx(self, i_real, j_real, Lat_intra, Lon_intra):
        lat_l, lat_u = 39.92 - self.lat_step * i_real - self.lat_step / 2, 39.92 - self.lat_step * i_real + self.lat_step / 2
        lon_l, lon_u = -160 + self.lon_step * j_real - self.lon_step / 2, -160 + self.lon_step * j_real + self.lon_step / 2
        current_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u) & (Lon_intra > lon_l) & (Lon_intra <= lon_u)
        return current_idx

    def get_current_idx_fast(self, i_real, j_real, Lat_intra, Lon_intra):
        lat_l, lat_u = 39.92 - self.lat_step * i_real - self.lat_step / 2, 39.92 - self.lat_step * i_real + self.lat_step / 2
        lon_l, lon_u = -160 + self.lon_step * j_real - self.lon_step / 2, -160 + self.lon_step * j_real + self.lon_step / 2
        
        current_idx = np.full(Lat_intra.shape, False)
        sub_idx = (Lat_intra >= lat_l) & (Lat_intra < lat_u)
        current_idx[sub_idx] =  (Lon_intra[sub_idx] > lon_l) & (Lon_intra[sub_idx] <= lon_u)
    
        return current_idx

    def path_adj_reward_new(self, node_i, node_j, T0, Height_t, alpha_list=[0.2]):

            # "This function is not well-maintained! It use 1st order expansion with Q. \nTry self.algo_best_adj_path_new()")
        T_plan = len(node_i)
        n_alpha = len(alpha_list)
        R_seq_raw = -np.ones(T_plan)
        R_seq_adj_1 = -np.ones((T_plan, n_alpha))
        particle_list_dict = {}

        wave_6_list = -np.ones(T_plan)
        for t in range(T_plan):
            day, step = (T0 + t) // 8, (T0 + t) % 8
            Lat_intra = (self.Lat_matrix[:, day-self.start_day] * (8 - step) + step * self.Lat_matrix[:, day-self.start_day + 1]) / 8
            Lon_intra = (self.Lon_matrix[:, day-self.start_day] * (8 - step) + step * self.Lon_matrix[:, day-self.start_day + 1]) / 8

            # Location of visit cell:

            i_real, j_real = node_i[t], node_j[t]

            wave_height = Height_t[i_real, j_real, T0 + t-8*self.start_day]
            wave_6_list[t] = wave_height <= 6
            if wave_6_list[t]:
                particle_list_dict[t] = self.get_current_idx(i_real, j_real, Lat_intra, Lon_intra)
                R_seq_raw[t] = np.sum(particle_list_dict[t])

                delta_r_1 = 0

                for tt in range(t):
                    if wave_6_list[tt]:
                        r_inter_1 = np.sum(particle_list_dict[tt] & particle_list_dict[t])
                        delta_r_1 += r_inter_1
                R_seq_adj_1[t, :] = np.maximum(0, R_seq_raw[t] - delta_r_1 * np.array(alpha_list))
            else:
                R_seq_raw[t] = 0
                R_seq_adj_1[t, :] = 0

        return R_seq_raw, R_seq_adj_1

    @staticmethod
    def get_span_weight(delta_t, span_rule):
    ## return the weight of alpha for a given delta_t (2/speed) and span_rule
        if span_rule == "square":
            return delta_t**2 / 4
        elif span_rule == "linear":
            return delta_t / 2
        elif span_rule == "constant":
            return 1
        else:
            raise ValueError("span_rule should be square, linear or constant")