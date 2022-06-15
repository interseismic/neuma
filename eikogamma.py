from torch import Tensor
import torch
import torch.nn.functional as F
from pyproj import Proj
import time

import copy
import json

from math import *

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use('Agg')

pd.options.mode.chained_assignment = None  # default='warn'

class HuberDensity():
    def __init__(self, σ_T, M):
        self.loss = torch.nn.HuberLoss(delta=1.345/0.6745, reduction='none')
        self.M = M
    def log_prob(self, z):
        logL = self.loss(z/self.M, torch.zeros(z.shape, device=z.device))
        return -logL

def IO_JSON(file, Events=None, rw_type='r'):
    '''
        Reading/Writing in JSON file into location archieve
    '''
    if rw_type == 'w':
        tmpEvents = copy.deepcopy(Events)
    elif rw_type == 'a+':
        tmpEvents = copy.deepcopy(Events)
    elif rw_type == 'r':
        with open(file, 'r') as f:
            tmpEvents = json.load(f)

    for key in tmpEvents.keys():
        if rw_type == 'w':
            tmpEvents[key]['Picks'] = tmpEvents[key]['Picks'].astype(
                str).to_dict()
        elif rw_type == 'a+':
            tmpEvents[key]['Picks'] = tmpEvents[key]['Picks'].astype(
                str).to_dict()
        elif rw_type == 'r':
            tmpEvents[key]['Picks'] = pd.DataFrame.from_dict(
                tmpEvents[key]['Picks'])
        else:
            print(
                'Please specify either "read (r)", "write (w)" or "append (a)" for handelling the data')

    if rw_type == 'w':
        with open(file, rw_type) as f:
            json.dump(tmpEvents, f)
        del tmpEvents
    elif rw_type == 'a+':
        try:
            with open(file, 'r+') as f:
                d = json.load(f)
                d.update(tmpEvents)
                f.seek(0)
        except BaseException:
            d = tmpEvents
            print('Creating Appending Catalog - {}'.format(file))
        with open(file, 'w') as f:
            json.dump(d, f)
        del tmpEvents, d
    elif rw_type == 'r':
        return tmpEvents

# =========== INPUT/OUTPUT FORMAT =======


def IO_NLLoc2JSON(file, EVT={}, startEventID=1000000):
    # Reading in the lines
    f = open(file, "r")
    lines = f.readlines()
    lds = np.where(np.array(lines) == '\n')[
        0] - np.arange(len(np.where(np.array(lines) == '\n')[0]))
    lines_start = np.append([0], lds[:-1])
    lines_end = lds

    # Reading in the event lines
    evt = pd.read_csv(
        file,
        sep=r'\s+',
        names=[
            'Station',
            'Network',
            'r1',
            'r2',
            'PhasePick',
            'r3',
            'Date',
            'Time',
            'Sec',
            'r4',
            'PickError',
            'r5',
            'r6',
            'r7'])
    evt['DT'] = pd.to_datetime(evt['Date'].astype(str).str.slice(stop=4) + '/' +
                               evt['Date'].astype(str).str.slice(start=4, stop=6) + '/' +
                               evt['Date'].astype(str).str.slice(start=6, stop=8) + 'T' +
                               evt['Time'].astype(str).str.zfill(4).str.slice(stop=2) + ':' +
                               evt['Time'].astype(str).str.zfill(4).str.slice(start=2) + ':' +
                               evt['Sec'].astype(str).str.split('.', expand=True)[0].str.zfill(2) + '.' +
                               evt['Sec'].astype(str).str.split('.', expand=True)[1].str.zfill(2), format='%Y/%m/%dT%H:%M:%S.%f')
    evt = evt[['Network', 'Station', 'PhasePick', 'DT', 'PickError']]

    # Turning
    for eds in range(len(lines_start)):
        evt_tmp = evt.iloc[lines_start[eds]:lines_end[eds]]
        EVT['{}'.format(startEventID + eds)] = {}
        EVT['{}'.format(startEventID + eds)
            ]['Picks'] = evt_tmp.reset_index(drop=True)

    return EVT


def IO_JSON2CSV(EVT, savefile=None):
    '''
        Saving Events in CSV format
    '''

    Events = EVT

    # Loading location information
    picks = (np.zeros((len(Events.keys()), 8)) * np.nan).astype(str)
    for indx, evtid in enumerate(Events.keys()):
        try:
            picks[indx, 0] = str(evtid)
            picks[indx, 1] = Events[evtid]['location']['OriginTime']
            picks[indx, 2:5] = (
                np.array(Events[evtid]['location']['Hypocenter'])).astype(str)
            picks[indx, 5:] = (
                np.array(Events[evtid]['location']['HypocenterError'])).astype(str)
        except BaseException:
            continue
    picks_df = pd.DataFrame(picks,
                            columns=['EventID', 'DT', 'X', 'Y', 'Z', 'ErrX', 'ErrY', 'ErrZ'])
    picks_df['X'] = picks_df['X'].astype(float)
    picks_df['Y'] = picks_df['Y'].astype(float)
    picks_df['Z'] = picks_df['Z'].astype(float)
    picks_df['ErrX'] = picks_df['ErrX'].astype(float)
    picks_df['ErrY'] = picks_df['ErrY'].astype(float)
    picks_df['ErrZ'] = picks_df['ErrZ'].astype(float)
    picks_df = picks_df.dropna(axis=0)
    picks_df['DT'] = pd.to_datetime(picks_df['DT'])
    picks_df = picks_df[['EventID', 'DT', 'X',
                         'Y', 'Z', 'ErrX', 'ErrY', 'ErrZ']]

    if isinstance(savefile, type(None)):
        return picks_df
    else:
        picks_df.to_csv(savefile, index=False)


# =========== MAIN =======
class EikoGMM(torch.nn.Module):
    def __init__(self, EikoNet, Phases=[
                 'P', 'S'], device='cpu', lr=1, n_clusters=150):
        super(EikoGMM, self).__init__()

        # -- Defining the EikoNet input formats
        self.eikonet_Phases = Phases
        self.eikonet_models = EikoNet
        if len(self.eikonet_Phases) != len(self.eikonet_models):
            print('Error - Number of phases not equal to number of EikoNet models')

        # Determining if the EikoNets are solved for the same domain
        xmin_stack = np.vstack(
            [self.eikonet_models[x].Params['VelocityClass'].xmin for x in range(len(self.eikonet_models))])
        xmax_stack = np.vstack(
            [self.eikonet_models[x].Params['VelocityClass'].xmax for x in range(len(self.eikonet_models))])
        if not (xmin_stack == xmin_stack[0, :]).all() or not (
                xmax_stack == xmax_stack[0, :]).all():
            print(
                'Error - EikoNet Models not in the same domain\n Min Points = {}\n Max Points = {}'.format(
                    xmin_stack,
                    xmax_stack))

        self.VelocityClass = self.eikonet_models[0].Params['VelocityClass']

        # Converting to UTM projection scheme form
        self.proj_str = copy.copy(
            self.eikonet_models[0].Params['VelocityClass'].projection)
        if not isinstance(self.proj_str, type(None)):
            self.projection = Proj(self.proj_str)
            self.xmin = copy.copy(self.VelocityClass.xmin)
            self.xmax = copy.copy(self.VelocityClass.xmax)
            self.xmin[0], self.xmin[1] = self.projection(
                self.xmin[0], self.xmin[1])
            self.xmax[0], self.xmax[1] = self.projection(
                self.xmax[0], self.xmax[1])
        else:
            self.projection = None
            self.xmin = copy.copy(self.VelocityClass.xmin)
            self.xmax = copy.copy(self.VelocityClass.xmax)

        self.device = torch.device(device)
        self.xmin = torch.tensor(self.xmin, device=self.device)
        self.xmax = torch.tensor(self.xmax, device=self.device)
        # --------- Initialising Location Information ---------
        # -- Defining the device to run the location procedure on

        # -- Defining the parameters required in the earthquake location procedure
        self.location_info = {}
        self.location_info['Likelihood'] = 'Laplace'
        self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'] = [0.0, 0.0, 0.0]
        self.location_info['Individual Event Epoch Save and Print Rate'] = [
            None, False]
        self.location_info['Number of samples'] = n_clusters
        self.location_info['lr'] = lr
        self.location_info['Save every * events'] = 100
        self.location_info['Location Uncertainty Percentile (%)'] = 90.0

        # Depricated values
        self.location_info['Hypocenter Cluster - Seperation (km)'] = None
        self.location_info['Hypocenter Cluster - Minimum Samples'] = None

        # --------- Initialising Plotting Information ---------
        self.plot_info = {}

        # - Event Plot parameters
        # Location plotting
        self.plot_info['EventPlot'] = {}
        self.plot_info['EventPlot']['Domain Distance'] = 50
        self.plot_info['EventPlot']['Save Type'] = 'png'
        self.plot_info['EventPlot']['Figure Size Scale'] = 1.0
        self.plot_info['EventPlot']['Plot kde'] = True
        self.plot_info['EventPlot']['NonClustered SVGD'] = [0.5, 'r']
        self.plot_info['EventPlot']['Clustered SVGD'] = [1.2, 'g']
        self.plot_info['EventPlot']['Hypocenter Location'] = [15, 'k']
        self.plot_info['EventPlot']['Hypocenter Errorbar'] = [True, 'k']
        self.plot_info['EventPlot']['Legend'] = True

        # Optional Station Plotting
        self.plot_info['EventPlot']['Stations'] = {}
        self.plot_info['EventPlot']['Stations']['Plot Stations'] = True
        self.plot_info['EventPlot']['Stations']['Station Names'] = True
        self.plot_info['EventPlot']['Stations']['Marker Color'] = 'b'
        self.plot_info['EventPlot']['Stations']['Marker Size'] = 25

        # Optional Trace Plotting
        self.plot_info['EventPlot']['Traces'] = {}
        self.plot_info['EventPlot']['Traces']['Plot Traces'] = False
        self.plot_info['EventPlot']['Traces']['Trace Host'] = None
        self.plot_info['EventPlot']['Traces']['Trace Host Type'] = '/YEAR/JD/*ST*'
        self.plot_info['EventPlot']['Traces']['Channel Types'] = ['EH*', 'HH*']
        self.plot_info['EventPlot']['Traces']['Filter Freq'] = [2, 16]
        self.plot_info['EventPlot']['Traces']['Normalisation Factor'] = 1.0
        self.plot_info['EventPlot']['Traces']['Time Bounds'] = [0, 5]
        self.plot_info['EventPlot']['Traces']['Pick linewidth'] = 2.0
        self.plot_info['EventPlot']['Traces']['Trace linewidth'] = 1.0

        # - Catalog Plot parameters
        self.plot_info['CatalogPlot'] = {}
        self.plot_info['CatalogPlot']['Minimum Phase Picks'] = 12
        self.plot_info['CatalogPlot']['Maximum Location Uncertainty (km)'] = 15
        self.plot_info['CatalogPlot']['Event Info - [Size, Color, Marker, Alpha]'] = [
            0.1, 'r', '*', 0.8]
        self.plot_info['CatalogPlot']['Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]'] = [
            True, 0.1, 'r', 0.8]
        self.plot_info['CatalogPlot']['Station Marker - [Size,Color,Names On/Off(Bool)]'] = [
            15, 'b', True]
        self.plot_info['CatalogPlot']['Fault Planes - [Size,Color,Marker,Alpha]'] = [
            0.1, 'gray', '-', 1.0]

        # --- Variables that are updated in run-time
        self._σ_T = None
        self._optimizer = None
        self._orgTime = None

    def locVar(self, T_obs, T_obs_err):
        '''
            Applying variance from Pick and Distance weighting to each of the observtions
        '''
        # Intialising a variance of the LOCGAU2 settings
        self._σ_T = torch.clamp(T_obs * self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'][0],
                                self.location_info[
            'Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'][1],
            self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'][2]).to(self.device)**2
        # Adding the variance of the Station Pick Uncertainties
        self._σ_T += (T_obs_err**2)
        # Turning back into a std
        self._σ_T = torch.sqrt(self._σ_T)

    def compute_dtimes_obs(self, t_obs, t_obs_err, t_phase, n_clusters):

        # Determining the predicted Travel-time for the different phases
        n_obs = 0
        cc = 0
        for ind, phase in enumerate(self.eikonet_Phases):
            if phase == 'P':
                phs = 0
            else:
                phs = 1
            phase_index = (t_phase == phs).nonzero(as_tuple=True)[0]
            if len(phase_index) == 0:
                continue

            pha_T_obs = t_obs[phase_index].repeat(n_clusters, 1)
            pha_T_obs_err = t_obs_err[phase_index].repeat(n_clusters, 1)
            if cc == 0:
                n_obs = len(phase_index)
                T_obs = pha_T_obs
                T_obs_err = pha_T_obs_err
                cc += 1
            else:
                n_obs += len(phase_index)
                T_obs = torch.cat([T_obs, pha_T_obs], dim=1)
                T_obs_err = torch.cat([T_obs_err, pha_T_obs_err], dim=1)

        self._σ_T = T_obs_err
        return T_obs

    def compute_dtimes_syn(self, X_src, X_rec, T_phase):
        # Preparing EikoNet input
        n_clusters = X_src.shape[0]
        cc = 0
        for ind, phase in enumerate(self.eikonet_Phases):
            if phase == 'P':
                phs = 0
            else:
                phs = 1
            phase_index = (T_phase == phs).nonzero(as_tuple=True)[0]
            if len(phase_index) == 0:
                continue

            pha_X_inp = torch.cat([X_src[:,:3].repeat_interleave(
                len(phase_index), dim=0), X_rec[phase_index, :].repeat(n_clusters, 1)], dim=1)
            pha_T_pred = self.eikonet_models[ind].TravelTimes(
                pha_X_inp, projection=False).reshape(
                n_clusters, len(phase_index))

            pha_T_pred += X_src[:,3].unsqueeze(1)
            if cc == 0:
                n_obs = len(phase_index)
                T_pred = pha_T_pred
                cc += 1
            else:
                n_obs += len(phase_index)
                T_pred = torch.cat([T_pred, pha_T_pred], dim=1)

        #dT_pred = T_pred[:, self.pairs[:, 0]] - T_pred[:, self.pairs[:, 1]]
        return T_pred

    def _compute_origin(self, T_obs, t_phase, X_rec, Hyp):
        '''
            Internal function to compute origin time and predicted Travel-times from Obs and Predicted Travel-times
        '''

        # Determining the predicted Travel-time for the different phases
        cc = 0
        for ind, phase in enumerate(self.eikonet_Phases):
            if phase == 'P':
                phs = 0
            else:
                phs = 1
            phase_index = (t_phase == phs).nonzero(as_tuple=True)[0]

            if len(phase_index) != 0:
                pha_X_inp = torch.cat([torch.repeat_interleave(
                    Hyp[None, :], len(phase_index), dim=0), X_rec[phase_index, :]], dim=1)
                pha_T_obs = T_obs[phase_index]

                # pha_X_inp.requires_grad_()
                pha_T_pred = self.eikonet_models[ind].TravelTimes(
                    pha_X_inp, projection=False)

                # # -- Determining take-off angles --
                if cc == 0:
                    T_obs_tmp = pha_T_obs
                    T_pred = pha_T_pred
                    phase_idx_full = phase_index
                    cc += 1
                else:
                    T_obs_tmp = torch.cat([T_obs_tmp, pha_T_obs])
                    T_pred = torch.cat([T_pred, pha_T_pred])
                    phase_idx_full = torch.cat([phase_idx_full, phase_index])

        OT = np.median((T_pred - T_obs_tmp).detach().cpu().numpy())
        pick_TD = ((T_pred - OT) - T_obs_tmp).detach().cpu().numpy()
        OT_std = np.nanmedian(abs(pick_TD))
        
        pick_TD_new = np.zeros(pick_TD.size)
        for i in range(pick_TD.size):
            pick_TD_new[phase_idx_full[i]] = pick_TD[i]
        pick_TD = pick_TD_new

        return OT, OT_std, pick_TD

    def SyntheticCatalog(self, input_file, Stations, save_file=None):
        '''
            Determining synthetic Travel-times between source and reciever locations, returning a JSON pick file for each event


            Event_Locations - EventNum, OriginTime, PickErr, X, Y, Z

            Stations -

            # JDS - MAKE CORRECTIONS TO PROJECTION !!


        '''

        # Determining the predicted Travel-time to each of the stations to corresponding
        # source locations. Optional argumenent to return them as json pick
        evtdf = pd.read_csv(input_file)
        EVT = {}
        for indx in range(len(evtdf)):
            EVT['{}'.format(evtdf['EventNum'].iloc[indx])] = {}

            OT = evtdf['OriginTime'].iloc[indx]

            # Defining the picks to append
            picks = pd.DataFrame(
                columns=[
                    'Network',
                    'Station',
                    'PhasePick',
                    'DT',
                    'PickError'])
            for ind, phs in enumerate(self.eikonet_Phases):
                picks_phs = Stations[['Network', 'Station', 'X', 'Y', 'Z']]
                picks_phs['PhasePick'] = phs
                picks_phs['PickError'] = evtdf['PickErr'].iloc[indx]
                Pairs = np.zeros((int(len(Stations)), 6))
                Pairs[:, :3] = np.array(evtdf[['X', 'Y', 'Z']].iloc[indx])
                Pairs[:, 3:] = np.array(picks_phs[['X', 'Y', 'Z']])

                if not isinstance(self.projection, type(None)):
                    Pairs[:, 0], Pairs[:, 1] = self.projection(
                        Pairs[:, 0], Pairs[:, 1])
                    Pairs[:, 3], Pairs[:, 4] = self.projection(
                        Pairs[:, 3], Pairs[:, 4])

                Pairs = Tensor(Pairs)

                Pairs = Pairs.to(self.device)
                TT_pred = self.eikonet_models[ind].TravelTimes(
                    Pairs, projection=False).detach().to('cpu').numpy()
                del Pairs

                picks_phs['DT'] = TT_pred
                picks_phs['DT'] = (
                    pd.to_datetime(OT) +
                    pd.to_timedelta(
                        picks_phs['DT'],
                        unit='S')).dt.strftime('%Y/%m/%dT%H:%M:%S.%f')

                picks = picks.append(
                    picks_phs[['Network', 'Station', 'PhasePick', 'DT', 'PickError']])

            EVT['{}'.format(evtdf['EventNum'].iloc[indx])]['Picks'] = picks

        if isinstance(save_file, str):
            IO_JSON('{}.json'.format(save_file), Events=EVT, rw_type='w')

        return EVT

    def Events2CSV(self, EVT=None, savefile=None, projection=None):
        '''
            Saving Events in CSV format
        '''

        if isinstance(EVT, type(None)):
            Events = self.Events
        else:
            Events = EVT

        # Loading location information
        picks = (np.zeros((len(Events.keys()), 8)) * np.nan).astype(str)
        for indx, evtid in enumerate(Events.keys()):
            try:
                picks[indx, 0] = str(evtid)
                picks[indx, 1] = self.Events[evtid]['location']['OriginTime']
                picks[indx, 2:5] = (
                    np.array(self.Events[evtid]['location']['Hypocenter'])).astype(str)
                picks[indx, 5:] = (
                    np.array(self.Events[evtid]['location']['HypocenterError'])).astype(str)
            except BaseException:
                continue
        picks_df = pd.DataFrame(picks,
                                columns=['EventID', 'DT', 'X', 'Y', 'Z', 'ErrX', 'ErrY', 'ErrZ'])
        picks_df['X'] = picks_df['X'].astype(float)
        picks_df['Y'] = picks_df['Y'].astype(float)
        picks_df['Z'] = picks_df['Z'].astype(float)
        picks_df['ErrX'] = picks_df['ErrX'].astype(float)
        picks_df['ErrY'] = picks_df['ErrY'].astype(float)
        picks_df['ErrZ'] = picks_df['ErrZ'].astype(float)
        picks_df = picks_df.dropna(axis=0)
        picks_df['DT'] = pd.to_datetime(picks_df['DT'])
        picks_df = picks_df[['EventID', 'DT', 'X',
                             'Y', 'Z', 'ErrX', 'ErrY', 'ErrZ']]

        if isinstance(savefile, type(None)):
            return picks_df
        else:
            picks_df.to_csv(savefile, index=False)

    def train(self, T_obs, X_rec, T_obs_phase, generator, epochs, patience, early_stopping, n_clusters):
        losses = []
        optimizer = torch.optim.Adam(generator.parameters(), lr = self.location_info['lr'])
        for epoch in range(epochs):

            x_samples_transformed, logdet = generator.reverse(torch.randn((n_clusters, 4), device=self.device))

            #X_src = torch.sigmoid(x_samples_transformed)
            X_src = x_samples_transformed
            X_src[:,:3] = torch.sigmoid(X_src[:,:3])
            X_src[:,:3] = X_src[:,:3] * (self.xmax - self.xmin) + self.xmin
            det_sigmoid = torch.sum(-x_samples_transformed - 2 * torch.nn.Softplus()(-x_samples_transformed), -1)
            logdet = logdet + det_sigmoid

            T_syn = self.compute_dtimes_syn(X_src, X_rec, T_obs_phase)
            #dT_syn = self.compute_dtimes_syn(X_src, X_rec, T_obs_phase)
            loss = torch.mean(-self.log_L.log_prob(T_obs-T_syn).sum(dim=1) - logdet)
            losses += [loss.item()]

            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(generator.parameters(), 1e-5)
            optimizer.step()

            if (epoch + 1) % 30 == 0:
                with torch.no_grad():
                    print(f"epoch: {epoch:}, loss: {loss.item():.5f}", "{}".format(X_src.mean(dim=0)))

            resid = (T_obs - T_syn).mean(dim=0)

            if early_stopping == True:
                if epoch <= patience:
                    continue
                best_before = np.min(losses[:(epoch - patience)])
                best_after = np.min(losses[(epoch - patience):])
                if best_before < best_after:
                    print(
                        "Reached early stopping criteria at epoch {} loss {}".format(epoch, best_before))
                    break
        return losses, resid

    def laplace_pdf(self, x, b):
        return torch.exp(-torch.abs(x)/b) / (2*b)

    def uniform_pdf(self, x, a, b):
        # return torch.where((x >= a) & (x <= b), float(1.0 / (b-a)), float(0.0))
        return torch.zeros(x.shape, device=self.device)

    def MAP(self, K, T_obs, X_rec, T_obs_phase, epochs, patience, early_stopping):
        losses = []
        T0 = T_obs.min()
        T1 = T_obs.max()
        N = T_obs.shape[1]
        X_src = torch.zeros(K, 4, device=self.device)
        for i in range(3):
            X_src[:,i] = (self.xmax[i] - self.xmin[i])*torch.rand(K, device=self.device) + self.xmin[i]
        X_src[:,3] = (T1-T0) * torch.rand(K, device=self.device) + T0
        X_src = X_src.requires_grad_()
        best_src = None
        best_loss = 99999999.
        logit_phi = torch.zeros(K, device=self.device).requires_grad_()
        logit_w = torch.zeros(N, device=self.device).requires_grad_()
        logit_gamma = torch.zeros(K, N, device=self.device).requires_grad_()
        optimizer = torch.optim.Adam([X_src, logit_phi, logit_w, logit_gamma], lr = self.location_info['lr'])

        for epoch in range(epochs):
            optimizer.zero_grad()
            # T_syn is [K, N]
            T_syn = self.compute_dtimes_syn(X_src, X_rec, T_obs_phase)
            
            phi = F.softmax(logit_phi, -1)
            w = torch.sigmoid(logit_w)
            gamma = F.softmax(logit_gamma, dim=0)

            p_x_G = (gamma * phi[:,None] * self.laplace_pdf(T_obs-T_syn, 0.3)).sum(0)
            l_reg = 
            loss = -torch.log((1-w) * self.uniform_pdf(T_obs, T0, T1) + w * p_x_G).sum()

            losses += [loss.item()]

            loss.backward()
            # print()
            # print(logit_w.grad)
            # print(w)
            optimizer.step()

            if (epoch + 1) % 30 == 0:
                with torch.no_grad():
                    print(f"epoch: {epoch:}, loss: {loss.item():.5f}")
                    print(gamma.sum(1))

            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_src = X_src.detach()
                resid = T_obs-T_syn
            if early_stopping == True:
                if epoch <= patience:
                    continue
                best_before = np.min(losses[:(epoch - patience)])
                best_after = np.min(losses[(epoch - patience):])
                if best_before < best_after:
                    print(
                        "Reached early stopping criteria at epoch {} loss {}".format(epoch, best_before))
                    break

        return losses, best_src, resid

    def LocateEvents(self, EVTS, Stations, output_path, catalog_file, epochs=175, output_plots=False, timer=False,
                     PriorCatalog=False, early_stopping=False, patience=10, n_flow=16, seqfrac=0.5, M=None):
        from itertools import combinations
        self.Events = EVTS

        if PriorCatalog == False:
            try:
                os.system('rm {}/{}'.format(output_path, catalog_file))
            except BaseException:
                print

        if PriorCatalog == False:
            try:
                os.system('rm {}/{}'.format(output_path, catalog_file))
            except BaseException:
                print('      No Prior Catalog defined for appending')

        evtid = []
        for c, ev in enumerate(self.Events.keys()):
            # try:
            if timer == True:
                timer_start = time.time()

            evtid.append(ev)
            Ev = self.Events[ev]
            Ev['Picks'] = Ev['Picks'][['Network', 'Station', 'PhasePick', 'DT', 'PickError']]

            # Formating the pandas datatypes
            Ev['Picks']['Network'] = Ev['Picks']['Network'].astype(str)
            Ev['Picks']['Station'] = Ev['Picks']['Station'].astype(str)
            Ev['Picks']['PhasePick'] = Ev['Picks']['PhasePick'].astype(str)
            Ev['Picks']['DT'] = pd.to_datetime(Ev['Picks']['DT'])
            Ev['Picks']['PickError'] = Ev['Picks']['PickError'].astype(float)
            Ev['Picks'] = Ev['Picks'][Ev['Picks']['PhasePick'].isin(
                self.eikonet_Phases)].reset_index(drop=True)

            if len(Ev['Picks']) == 0:
                print('No phase picks ! Event cannot be located')
                continue

            # printing the current event being run
            print()
            t_start = time.time()
            print('================= Processing Event:{} - Event {} of {} - Number of observations={} =============='.format(
                ev, c + 1, len(self.Events.keys()), len(Ev['Picks'])))

            # Adding the station location to the pick files
            pick_info = pd.merge(
                Ev['Picks'], Stations[['Network', 'Station', 'X', 'Y', 'Z']])
            Ev['Picks'] = pick_info[['Network', 'Station', 'X', 'Y', 'Z', 'PhasePick', 'DT', 'PickError']]

            # Defining the arrivals times in seconds
            pick_info['Seconds'] = (pick_info['DT'] - np.min(pick_info['DT'])).dt.total_seconds()

            # Applying projection
            X_rec = np.array(pick_info[['X', 'Y', 'Z']])
            if not isinstance(self.projection, type(None)):
                X_rec[:, 0], X_rec[:, 1] = self.projection(
                    X_rec[:, 0], X_rec[:, 1])

            X_rec = Tensor(X_rec).to(self.device)
            T_obs = Tensor(np.array(pick_info['Seconds'])).to(self.device)
            T_obs_err = Tensor(np.array(pick_info['PickError'])).to(self.device)
            T_obs_phase = [0 if x == 'P' else 1 for x in pick_info['PhasePick']]
            T_obs_phase = torch.tensor(T_obs_phase).to(self.device)

            n_clusters = int(self.location_info['Number of samples'])
            T_obs = self.compute_dtimes_obs(T_obs, T_obs_err, T_obs_phase, n_clusters)

            if self.location_info['Likelihood'] == 'Normal':
                self.log_L = torch.distributions.Normal(torch.zeros(T_obs.shape[1], device=self.device), self._σ_T)
            elif self.location_info['Likelihood'] == 'Laplace':
                self.log_L = torch.distributions.Laplace(torch.zeros(T_obs.shape[1], device=self.device), self._σ_T)
                self.loss = torch.nn.L1Loss()
            elif self.location_info['Likelihood'] == 'Cauchy':
                self.log_L = torch.distributions.Cauchy(torch.zeros(T_obs.shape[1], device=self.device), self._σ_T)
            elif self.location_info['Likelihood'] == 'Huber':
                if M is None:
                    print("M must be defined for Huber loss")
                    return
                self.log_L = HuberDensity(self._σ_T, M)
                self.loss = torch.nn.HuberLoss(delta=M)
            else:
                self.log_L = None
                self.loss = None

            losses, X_src, resid = self.MAP(n_clusters, T_obs, X_rec, T_obs_phase, epochs, patience, early_stopping)

            # return
            continue

            Ev['location'] = {}
            Ev['location']['SVGD_points'] = X_src.detach().cpu().numpy().tolist()
            Ev['location']['loss_at_epoch'] = losses

            pts = np.transpose(X_src[:,:3].detach().cpu().numpy())
            hyp = np.mean(pts, axis=1)
            err = 0.5 * np.abs(np.percentile(pts, 95, axis=1) - np.percentile(pts, 5, axis=1))

            Ev['location']['Hypocenter'] = (hyp).tolist()
            Ev['location']['HypocenterError'] = np.array(
                [err[0], err[1], err[2]]).tolist()

            # pick_TD = np.zeros(X_rec.shape[0])
            originOffset = torch.median(X_src[:,3])
            originOffset_std = 0.5 * np.abs(np.percentile(X_src[:,3].detach().cpu().numpy(), 95) - 
                                     np.percentile(X_src[:,3].detach().cpu().numpy(), 5))
            Ev['location']['OriginTime_std'] = float(originOffset_std)
            Ev['location']['OriginTime'] = str(np.min(pick_info['DT']) - pd.Timedelta(float(originOffset), unit='S'))
            Ev['Picks']['TimeDiff'] = resid.squeeze().detach().cpu().numpy()

            # -- Applying the projection from UTM to LatLong
            if not isinstance(self.projection, type(None)):
                Ev['location']['Hypocenter'] = np.array(
                    Ev['location']['Hypocenter'])
                Ev['location']['Hypocenter'][0], Ev['location']['Hypocenter'][1] = self.projection(
                    Ev['location']['Hypocenter'][0], Ev['location']['Hypocenter'][1], inverse=True)
                Ev['location']['Hypocenter'] = Ev['location']['Hypocenter'].tolist()

                Ev['location']['SVGD_points'] = np.array(
                    Ev['location']['SVGD_points'])
                Ev['location']['SVGD_points'][:, 0], Ev['location']['SVGD_points'][:, 1] = self.projection(
                    Ev['location']['SVGD_points'][:, 0], Ev['location']['SVGD_points'][:, 1], inverse=True)
                Ev['location']['SVGD_points'] = Ev['location']['SVGD_points'].tolist()

            print('---- OT= {} +/- {}s - Hyp=[{:.2f}, {:.2f}, {:.2f}] - Hyp Uncertainty (+/- km)=[{:.2f}, {:.2f}, {:.2f}]'.format(
                Ev['location']['OriginTime'], Ev['location']['OriginTime_std'], Ev['location']['Hypocenter'][0],
                Ev['location']['Hypocenter'][1], Ev['location']['Hypocenter'][2], Ev['location']['HypocenterError'][0],
                Ev['location']['HypocenterError'][1], Ev['location']['HypocenterError'][2]))

            if timer == True:
                timer_end = time.time()
                print('Processing took {}s'.format(timer_end - timer_start))

            # Plotting Event plots
            if output_plots:
                if timer == True:
                    timer_start = time.time()
                print('---- Saving Event Plot ----')
                self.EventPlot(output_path, Ev, EventID=ev)

                if timer == True:
                    timer_end = time.time()
                    print('Plotting took {}s'.format(timer_end - timer_start))

            # Saving Catalog instance
            if (self.location_info['Save every * events'] is not None) and (
                    (c % self.location_info['Save every * events']) == 0):
                if timer == True:
                    timer_start = time.time()
                print('---- Saving Catalog instance ----')
                IO_JSON('{}/{}'.format(output_path, catalog_file),
                    Events={ev: self.Events[ev] for ev in evtid}, rw_type='a+')
                {ev: 'Located & Saved' for ev in evtid}  # Freeing up memory
                if timer == True:
                    timer_end = time.time()
                    print('Saving took {}s'.format(timer_end - timer_start))

        # Writing out final Catalog
        IO_JSON('{}/{}'.format(output_path, catalog_file),
                Events={ev: self.Events[ev] for ev in evtid}, rw_type='a+')
        {ev: 'Located & Saved' for ev in evtid}

    def EventPlot(self, PATH, Event, EventID=None):
        plt.close('all')
        OT = str(Event['location']['OriginTime'])
        OT_std = Event['location']['OriginTime_std']
        locs = np.array(Event['location']['SVGD_points'])
        optimalloc = np.array(Event['location']['Hypocenter'])
        optimalloc_Err = np.array(Event['location']['HypocenterError'])
        Stations = Event['Picks'][['Station', 'X', 'Y', 'Z']]

        if self.plot_info['EventPlot']['Traces']['Plot Traces'] == True:
            fig = plt.figure(
                figsize=(
                    20 * self.plot_info['EventPlot']['Figure Size Scale'],
                    9 * self.plot_info['EventPlot']['Figure Size Scale']))
            xz = plt.subplot2grid((3, 5), (2, 0), colspan=2)
            xy = plt.subplot2grid(
                (3, 5), (0, 0), colspan=2, rowspan=2, sharex=xz)
            yz = plt.subplot2grid((3, 5), (0, 2), rowspan=2, sharey=xy)
            trc = plt.subplot2grid((3, 5), (0, 3), rowspan=3, colspan=2)
        else:
            fig = plt.figure(
                figsize=(
                    9 * self.plot_info['EventPlot']['Figure Size Scale'],
                    9 * self.plot_info['EventPlot']['Figure Size Scale']))
            xz = plt.subplot2grid((3, 3), (2, 0), colspan=2)
            xy = plt.subplot2grid(
                (3, 3), (0, 0), colspan=2, rowspan=2, sharex=xz)
            yz = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharey=xy)

        fig.patch.set_facecolor("white")

        # Specifying the label names
        xz.set_xlabel('UTM X (km)')
        xz.set_ylabel('Depth (km)')
        yz.set_ylabel('UTM Y (km)')
        yz.yaxis.tick_right()
        yz.yaxis.set_label_position("right")
        yz.set_xlabel('Depth (km)')

        if self.plot_info['EventPlot']['Domain Distance'] is not None:
            if not isinstance(self.projection, type(None)):
                optimalloc_UTM = copy.copy(optimalloc)
                optimalloc_UTM[0], optimalloc_UTM[1] = self.projection(
                    optimalloc_UTM[0], optimalloc_UTM[1])
                boundsmin = optimalloc_UTM - \
                    self.plot_info['EventPlot']['Domain Distance'] / 2
                boundsmax = optimalloc_UTM + \
                    self.plot_info['EventPlot']['Domain Distance'] / 2
                boundsmin[0], boundsmin[1] = self.projection(
                    boundsmin[0], boundsmin[1], inverse=True)
                boundsmax[0], boundsmax[1] = self.projection(
                    boundsmax[0], boundsmax[1], inverse=True)
            else:
                boundsmin = optimalloc - \
                    self.plot_info['EventPlot']['Domain Distance'] / 2
                boundsmax = optimalloc + \
                    self.plot_info['EventPlot']['Domain Distance'] / 2
            xy.set_xlim([boundsmin[0], boundsmax[0]])
            xy.set_ylim([boundsmin[1], boundsmax[1]])
            xz.set_xlim([boundsmin[0], boundsmax[0]])
            xz.set_ylim([boundsmin[2], boundsmax[2]])
            yz.set_xlim([boundsmin[2], boundsmax[2]])
            yz.set_ylim([boundsmin[1], boundsmax[1]])
        else:
            if not isinstance(self.projection, type(None)):
                lim_min = self.VelocityClass.xmin
                lim_max = self.VelocityClass.xmax
            else:
                lim_min = self.xmin
                lim_max = self.xmax
            xy.set_xlim([lim_min[0], lim_max[0]])
            xy.set_ylim([lim_min[1], lim_max[1]])
            xz.set_xlim([lim_min[0], lim_max[0]])
            xz.set_ylim([lim_min[2], lim_max[2]])
            yz.set_xlim([lim_min[2], lim_max[2]])
            yz.set_ylim([lim_min[1], lim_max[1]])

        # Invert yaxis
        xz.invert_yaxis()

        # Plotting the SVGD samples
        xy.scatter(locs[:, 0], locs[:, 1], float(self.plot_info['EventPlot']['NonClustered SVGD'][0]), str(
            self.plot_info['EventPlot']['NonClustered SVGD'][1]), label='SVGD Samples')

        # Plotting the predicted hypocentre and standard deviation location
        xy.scatter(
            optimalloc[0], optimalloc[1], float(
                self.plot_info['EventPlot']['Hypocenter Location'][0]), str(
                self.plot_info['EventPlot']['Hypocenter Location'][1]), label='Hypocenter')
        xz.scatter(
            optimalloc[0], optimalloc[2], float(
                self.plot_info['EventPlot']['Hypocenter Location'][0]), str(
                self.plot_info['EventPlot']['Hypocenter Location'][1]))
        yz.scatter(
            optimalloc[2], optimalloc[1], float(
                self.plot_info['EventPlot']['Hypocenter Location'][0]), str(
                self.plot_info['EventPlot']['Hypocenter Location'][1]))

        # Defining the Error bar location
        if self.plot_info['EventPlot']['Hypocenter Errorbar'][0]:

            # JDS - Currently these are rough errorbars, need to improve
            xy.errorbar(
                optimalloc[0],
                optimalloc[1],
                xerr=optimalloc_Err[0] / 111,
                yerr=optimalloc_Err[1] / 111,
                color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],
                label='Hyp {}% Confidence'.format(
                    self.location_info['Location Uncertainty Percentile (%)']))
            xz.errorbar(
                optimalloc[0],
                optimalloc[2],
                xerr=optimalloc_Err[0] / 111,
                yerr=optimalloc_Err[2],
                color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],
                label='Hyp {}% Confidence'.format(
                    self.location_info['Location Uncertainty Percentile (%)']))
            yz.errorbar(
                optimalloc[2],
                optimalloc[1],
                xerr=optimalloc_Err[2],
                yerr=optimalloc_Err[1] / 111,
                color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],
                label='Hyp {}% Confidence'.format(
                    self.location_info['Location Uncertainty Percentile (%)']))

        # Optional Station Location used in inversion
        if self.plot_info['EventPlot']['Stations']['Plot Stations']:
            idxsta = Stations['Station'].drop_duplicates().index
            station_markersize = self.plot_info['EventPlot']['Stations']['Marker Size']
            station_markercolor = self.plot_info['EventPlot']['Stations']['Marker Color']

            xy.scatter(Stations['X'].iloc[idxsta],
                       Stations['Y'].iloc[idxsta],
                       station_markersize, marker='^', color=station_markercolor, label='Stations')

            if self.plot_info['EventPlot']['Stations']['Station Names']:
                for i, txt in enumerate(Stations['Station'].iloc[idxsta]):
                    xy.annotate(
                        txt, (np.array(
                            Stations['X'].iloc[idxsta])[i], np.array(
                            Stations['Y'].iloc[idxsta])[i]))

            xz.scatter(Stations['X'].iloc[idxsta],
                       Stations['Z'].iloc[idxsta],
                       station_markersize, marker='^', color=station_markercolor)

            yz.scatter(Stations['Z'].iloc[idxsta],
                       Stations['Y'].iloc[idxsta],
                       station_markersize, marker='^', color=station_markercolor)

        # Defining the legend as top lef
        if self.plot_info['EventPlot']['Legend']:
            xy.legend(loc='upper left')
        plt.suptitle(
            ' Earthquake {} +/- {:.2f}s\n Hyp=[{:.2f},{:.2f},{:.2f}] - Hyp Uncertainty (km) +/- [{:.2f},{:.2f},{:.2f}]'.format(
                OT,
                OT_std,
                optimalloc[0],
                optimalloc[1],
                optimalloc[2],
                optimalloc_Err[0],
                optimalloc_Err[1],
                optimalloc_Err[2]))

        plt.savefig('{}/{}.{}'.format(PATH, EventID,
                    self.plot_info['EventPlot']['Save Type']))
        plt.clf()
        plt.close('all')

        plt.plot(Event['location']['loss_at_epoch'])
        plt.savefig('{}/{}_loss.{}'.format(PATH, EventID,
                    self.plot_info['EventPlot']['Save Type']))

    def CatalogPlot(self, filepath=None, Events=None, Stations=None, user_xmin=[
                      None, None, None], user_xmax=[None, None, None], Faults=None):

        if not isinstance(Events, type(None)):
            self.Events = Events

        # - Catalog Plot parameters
        min_phases = self.plot_info['CatalogPlot']['Minimum Phase Picks']
        max_uncertainty = self.plot_info['CatalogPlot'][
            'Maximum Location Uncertainty (km)']
        event_marker = self.plot_info['CatalogPlot']['Event Info - [Size, Color, Marker, Alpha]']
        event_errorbar_marker = self.plot_info['CatalogPlot'][
            'Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]']
        stations_plot = self.plot_info['CatalogPlot'][
            'Station Marker - [Size,Color,Names On/Off(Bool)]']
        fault_plane = self.plot_info['CatalogPlot']['Fault Planes - [Size,Color,Marker,Alpha]']

        fig = plt.figure(figsize=(15, 15))
        xz = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        xy = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, sharex=xz)
        yz = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharey=xy)

        # Defining the limits of the domain

        if not isinstance(self.projection, type(None)):
            lim_min = self.VelocityClass.xmin
            lim_max = self.VelocityClass.xmax
        else:
            lim_min = self.xmin
            lim_max = self.xmax

        for indx, val in enumerate(user_xmin):
            if val is not None:
                lim_min[indx] = val
        for indx, val in enumerate(user_xmax):
            if val is not None:
                lim_max[indx] = val

        xy.set_xlim([lim_min[0], lim_max[0]])
        xy.set_ylim([lim_min[1], lim_max[1]])
        xz.set_xlim([lim_min[0], lim_max[0]])
        xz.set_ylim([lim_min[2], lim_max[2]])
        yz.set_xlim([lim_min[2], lim_max[2]])
        yz.set_ylim([lim_min[1], lim_max[1]])

        # Specifying the label names
        xz.set_xlabel('UTM X (km)')
        xz.set_ylabel('Depth (km)')
        xz.invert_yaxis()
        yz.set_ylabel('UTM Y (km)')
        yz.yaxis.tick_right()
        yz.yaxis.set_label_position("right")
        yz.set_xlabel('Depth (km)')

        # Plotting the station locations
        if not isinstance(Stations, type(None)):
            sta = Stations[['Station', 'X', 'Y', 'Z']].drop_duplicates()
            xy.scatter(
                sta['X'],
                sta['Y'],
                stations_plot[0],
                marker='^',
                color=stations_plot[1],
                label='Stations')

            if stations_plot[2]:
                for i, txt in enumerate(sta['Station']):
                    xy.annotate(
                        txt, (np.array(
                            sta['X'])[i], np.array(
                            sta['Y'])[i]))

            xz.scatter(
                sta['X'],
                sta['Z'],
                stations_plot[0],
                marker='^',
                color=stations_plot[1])
            yz.scatter(
                sta['Z'],
                sta['Y'],
                stations_plot[0],
                marker='<',
                color=stations_plot[1])

        picks_df = self.Events2CSV()
        picks_df = picks_df[np.sum(
            picks_df[['ErrX', 'ErrY', 'ErrZ']], axis=1) <= max_uncertainty].reset_index(drop=True)

        xy.scatter(
            picks_df['X'],
            picks_df['Y'],
            event_marker[0],
            event_marker[1],
            marker=event_marker[2],
            alpha=event_marker[3],
            label='Catalog Locations')
        xz.scatter(
            picks_df['X'],
            picks_df['Z'],
            event_marker[0],
            event_marker[1],
            marker=event_marker[2],
            alpha=event_marker[3])
        yz.scatter(
            picks_df['Z'],
            picks_df['Y'],
            event_marker[0],
            event_marker[1],
            marker=event_marker[2],
            alpha=event_marker[3])

        # Plotting legend
        xy.legend(
            loc='upper left',
            markerscale=2,
            scatterpoints=1,
            fontsize=10)

        if filepath is not None:
            plt.savefig('{}'.format(filepath))
        else:
            plt.show()

        plt.close('all')
