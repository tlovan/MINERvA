import numpy as np
import ROOT
import os
import csv

def genTrackTimeOffset(source_dir, output_filename, scan_start, hist_bins, nbeams = 5, hist_max = 250):
    # Creating TChain from all files in source directory
    Chain = ROOT.TChain( "MasterAnaDev", "Chain" )
    nfiles = 0 

    for (root, dirs, files) in os.walk(f"{source_dir}/"):
        for file in files:
            if ("MasterAnaDev_data_AnaTuple_run" in file) and ("Playlist.root" in file):
                Chain.Add(f"{root}/{file}")
                nfiles += 1

    print(f"Chain created with {Chain.GetEntries()} entries from {nfiles} files.")

    # Setting only the relevant branches as active
    Chain.SetBranchStatus("*", False)
    Chain.SetBranchStatus("ev_gate", True)
    Chain.SetBranchStatus("vtx", True)
    Chain.SetBranchStatus("muon_trackVertexTime", True)
    
    # Histogram to which we can fit the beam structure
    tracktime_hist = ROOT.TH1D( "Track_Time_Hist", ";Track Time (ns)", hist_bins, 0, hist_max )

    # Not sure what these are for...
    prev_gate = 0
    prev_time = 0.

    # Calculating track time for each event
    for entry in Chain:        
        vtx_time = entry.muon_trackVertexTime - (entry.vtx[2] - 4000)/300.
        if entry.ev_gate == prev_gate:
            tracktime = vtx_time - prev_time
            if tracktime < hist_max:
                tracktime_hist.Fill(tracktime)

        prev_gate = entry.ev_gate
        prev_time = vtx_time

    bin_density = hist_bins/hist_max # bins per time

    fit_radius  = 6                              # radius about the mean beam time in ns to fit a gaussian curve
    beam_radius = round(fit_radius*bin_density)  # radius aobut the mean beam time in bins
    
    beam_period      = 18.831                          # beam periodicity in ns
    beam_period_bins = round(beam_period*bin_density)  # beam periodicity in bins  
    
    curr_time  = scan_start  # current time for the search

    beam_times = []
    start_times = []

    for beam_number in range(nbeams):
        fit = tracktime_hist.Fit("gaus", "S Quiet", "", curr_time-fit_radius, curr_time+fit_radius)
        time = fit.Parameter(1)
        start_time = time - (beam_number)*beam_period
        beam_times.append(time)
        start_times.append(start_time)

        curr_time += beam_period 

    mean_start_time = np.mean(start_times)
    
    output_file = open(f"{output_filename}_step.txt", 'w')
    output_file.write(f"{mean_start_time}")
    output_file.close()
    
def ParallelizeFiles(source_dir, output_filename, nsets):
    # Putting all files to process into a single list
    all_files = []

    for (root, dirs, files) in os.walk(f"{source_dir}/"):
        for file in files:
            if ("MasterAnaDev_data_AnaTuple_run" in file) and ("Playlist.root" in file):
                all_files.append(f"{root}/{file}")

    # Splitting the files into n equal sets
    file_sets = np.array_split(all_files, nsets)
    print(f"{nsets} file sets with length {len(file_sets[0])} created from {len(all_files)} files")

    # Writing file sets into an output file
    output_file = open(f"{output_filename}.txt", "w", newline = '')
    csvfile = csv.writer(output_file)
    
    for file_set in file_sets:
        csvfile.writerow(file_set)
    output_file.close()
    
# Function to get specific file set from a pre-generated list of file sets
def GetFileSet(fileset_name, set_number):
    input_file = open(f"{fileset_name}.txt", newline = '')
    csvfile = csv.reader(input_file)
    for i, row in enumerate(csvfile):
        if i == set_number:
            return row

def CreateDataTree(fileset_name, set_number):
    # Creating a TChain for all the files in the analysis
    Chain = ROOT.TChain( "MasterAnaDev", "Chain" )

    # Equivalent to the GetFileSet function
    input_file = open(f"{fileset_name}.txt", newline = '')
    csvfile = csv.reader(input_file)
    for i, row in enumerate(csvfile):
        if i == set_number:
            file_set = row

    # Looping through all the files in the source directory and any subfolders
    for file in file_set:
        Chain.AddFile(file)

    # To speed up processing, inactivating irrelevant branches
    Chain.SetBranchStatus("*", False)

    branch_list = {'ev_gate',
                   'ev_subrun',
                   'ev_run',
                   'vtx',
                   'muon_trackVertexTime', 
                   'MasterAnaDev_muon_E',
                   'MasterAnaDev_muon_P',
                   'MasterAnaDev_muon_Px',
                   'MasterAnaDev_muon_Py',
                   'MasterAnaDev_muon_Pz',
                   'MasterAnaDev_muon_theta', 
                   'MasterAnaDev_minos_trk_is_ok',
                   'MasterAnaDev_proton_E_fromdEdx',
                   'MasterAnaDev_proton_T_fromdEdx',
                   'MasterAnaDev_proton_P_fromdEdx',
                   'MasterAnaDev_proton_Px_fromdEdx',
                   'MasterAnaDev_proton_Py_fromdEdx',
                   'MasterAnaDev_proton_Pz_fromdEdx',
                   # 'MasterAnaDev_proton_theta_fromdEdX',
                   # 'n_minos_matches', 
                   # 'MasterAnaDev_in_fiducial_area',
                   'MasterAnaDev_pion_E',
                   'MasterAnaDev_pion_P',
                   'MasterAnaDev_pion_theta',  
                   # 'recoil_energy_nonmuon_vtx200mm'
                  }

    # Reactivating relevant branches
    for branch in branch_list:
        Chain.SetBranchStatus(branch, True)

    # Copying active data into a new file so it's easier to manipulate
    DataTree = Chain.CloneTree(0)
    DataTree.CopyEntries(Chain)
    
    return DataTree

def getLTMomenta(vector1, vector2):
    beam_correction = ROOT.Math.RotationX(-0.0575958653)

    v1_corr = beam_correction(vector1)
    v2_corr = beam_correction(vector2)

    v_combo = v1_corr + v2_corr
    
    v_transverse = np.sqrt( (v_combo.X())**2 + (v_combo.Y())**2 )
    v_longitude  = v_combo.Z()

    return v_transverse, v_longitude

def FileAnalysis(output_filename, fileset_name, set_number, beam_start_time):
    print(f"Beginning Analysis on file set {set_number}")
    print("--------------------------------------------")
    # Importing Data
    output_file = ROOT.TFile(f"{output_filename}{set_number}.root", 'recreate')
    data_tree = CreateDataTree(fileset_name, set_number)
    print("Data tree created")

    """ Defining Analysis Branches """
    "=========================================================="
    ana_tree = data_tree.CloneTree(0)
    
    track_time = np.array([0.]) # event track time (ns)
    beam_offset = np.array([0.]) # event offset from beam time (ns)
    transferred_mom = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # transferred 4mom |t| in muon-pion interactions [MeV^2]
    muon_proton_P_L = np.array([0.])
    muon_proton_P_T = np.array([0.])
    npions = np.array([0])
    
    ana_tree.Branch("track_time", track_time, "tracktime/D")
    ana_tree.Branch("beam_offset", beam_offset, "beam_offset/D")
    ana_tree.Branch("transferred_mom", transferred_mom, "transferred_mom/D")
    ana_tree.Branch("muon_proton_P_L", muon_proton_P_L, "muon_proton_P_L/D")
    ana_tree.Branch("muon_proton_P_T", muon_proton_P_T, "muon_proton_P_T/D")
    ana_tree.Branch("npions", npions, "npions")
    
    print("Analysis branches created")
    "=========================================================="

    """ Setting Branch Addresses """
    "=========================================================="
    ev_run    = np.array([0.])
    ev_subrun = np.array([0.])
    ev_gate   = np.array([0.])
    data_tree.SetBranchAddress("ev_run", ev_run)
    data_tree.SetBranchAddress("ev_subrun", ev_subrun)
    data_tree.SetBranchAddress("ev_gate", ev_gate)

    vtx         = np.zeros(4)
    minos_match = np.array([0.])
    muonTVT     = np.array([0.])
    # nparticles  = np.array([0] )
    data_tree.SetBranchAddress('vtx', vtx)
    data_tree.SetBranchAddress('MasterAnaDev_minos_trk_is_ok', minos_match)
    data_tree.SetBranchAddress('muon_trackVertexTime', muonTVT)
    # data_tree.SetBranchAddress('VertexTrackMultiplicity', nparticles)
    
    muonE     = np.array([0.])
    muonP     = np.array([0.])
    muonPx    = np.array([0.])
    muonPy    = np.array([0.])
    muonPz    = np.array([0.])
    muonTheta = np.array([0.])
    data_tree.SetBranchAddress('MasterAnaDev_muon_E', muonE)
    data_tree.SetBranchAddress('MasterAnaDev_muon_P', muonP)
    data_tree.SetBranchAddress('MasterAnaDev_muon_Px', muonPx)
    data_tree.SetBranchAddress('MasterAnaDev_muon_Py', muonPy)
    data_tree.SetBranchAddress('MasterAnaDev_muon_Pz', muonPz)
    data_tree.SetBranchAddress('MasterAnaDev_muon_theta', muonTheta)

    pionE     = np.zeros(10)
    pionP     = np.zeros(10)
    pionTheta = np.zeros(10)
    data_tree.SetBranchAddress('MasterAnaDev_pion_E', pionE)
    data_tree.SetBranchAddress('MasterAnaDev_pion_P', pionP)
    data_tree.SetBranchAddress('MasterAnaDev_pion_theta', pionTheta)

    protonT     = np.array([0.])
    protonP     = np.array([0.])
    protonPx    = np.array([0.])
    protonPy    = np.array([0.])
    protonPz    = np.array([0.])
    protonTheta = np.array([0.])
    data_tree.SetBranchAddress('MasterAnaDev_proton_T_fromdEdx', protonT)
    data_tree.SetBranchAddress('MasterAnaDev_proton_P_fromdEdx', protonP)
    data_tree.SetBranchAddress('MasterAnaDev_proton_Px_fromdEdx', protonPx)
    data_tree.SetBranchAddress('MasterAnaDev_proton_Py_fromdEdx', protonPy)
    data_tree.SetBranchAddress('MasterAnaDev_proton_Pz_fromdEdx', protonPz)
    print("Branch addresses set")
    "=========================================================="

    """ Analysis Loop """
    "=========================================================="
    beam_period = 18.831
    beam_size = beam_period/2 + 0.00001 # Adding a very small overlap to avoid the possibility of getting stuck in the beam offset loop, better way to prevent this? 

    N = data_tree.GetEntries()
    n = 0
    # Looping through all entries
    while n < N-1:
        # Checking for entries that have a match in run, subrun, and gate numbers
        data_tree.GetEntry(n+1)
        run1, subrun1, gate1 = ev_run[0], ev_subrun[0], ev_gate[0]
        time1 = muonTVT[0] - (vtx[2] - 4000)/300
        data_tree.GetEntry(n)
        run0, subrun0, gate0 = ev_run[0], ev_subrun[0], ev_gate[0]
        time0 = muonTVT[0] - (vtx[2] - 4000)/300
        """ Properties """
        # Script runs a bit differently depending on whether or not there is an gate, run subrun match
        if (run1 == run0) and (subrun1 == subrun0) and (gate1 == gate0):
            
            """ Both Events - Track Time and Beam Offset """
            "------------------------------------------------------"
            trk_time = np.abs(time0-time1)
            
            if trk_time < beam_start_time - beam_size: # Flagging any events before the start time (-1111)
                track_time[0]  = -1111
                beam_offset[0] = -1111
            elif trk_time > 344 and trk_time < 352: # Flagging the strange peak at ~348 ns (-4444)
                track_time[0]  = -4444
                beam_offset[0] = -4444
            else:
                track_time[0] = trk_time
                offset_time = trk_time - beam_start_time # Adjusting for the beam start time
                
                while offset_time > beam_size: # Finding the beam offset by subtracting intervals of the beam period
                    offset_time -= beam_period
                    
                beam_offset[0] = offset_time
            "------------------------------------------------------"

            """ First Event - Transferred 4 Momentum """
            "------------------------------------------------------"
            if minos_match == 1:
                muonP_T = muonP * np.sin(muonTheta)
                muonP_L = muonP * np.cos(muonTheta)

                for i, piE in enumerate(pionE):
                    if piE < 0: # Flagging invalid pions in array (-8888)
                        transferred_mom[i] = -8888
                    else:
                        npions[0] += 1
                        piP     = pionP[i]
                        piTheta = pionTheta[i]
                    
                        pionP_T = piP * np.sin(piTheta)
                        pionP_L = piP * np.sin(piTheta)

                        transferred_mom[i] = (piE + muonE - pionP_L - muonP_L)**2 + (pionP_T + muonP_T)**2
            else:
                for i, piE in enumerate(pionE):
                    if piE > 0:
                        npions[0] += 1
                    transferred_mom[i] = -7777 # Flagging non matched events (-7777)
            "------------------------------------------------------"

            """ First Event - Proton-Muon Transverse Momentum """
            "------------------------------------------------------"
            if (muonP != 0) and (protonP != 0):
                muonP_vec = ROOT.Math.XYZVector(muonPx, muonPy, muonPz)
                protonP_vec = ROOT.Math.XYZVector(protonPx, protonPy, protonPz)

                P_T, P_L = getLTMomenta(muonP_vec, protonP_vec)
                muon_proton_P_T[0] = P_T
                muon_proton_P_L[0] = P_L
            else:
                muon_proton_P_T[0] = -8888
                muon_proton_P_L[0] = -8888
            "------------------------------------------------------"

            """ Filling First Event """
            "------------------------------------------------------"
            ana_tree.Fill()
            "------------------------------------------------------"

            """ Second Event - Transferred 4 Momentum """
            "------------------------------------------------------"
            data_tree.GetEntry(n+1) # Setting the current entry to the second event
            if minos_match == 1:
                muonP_T = muonP * np.sin(muonTheta)
                muonP_L = muonP * np.cos(muonTheta)

                for i, piE in enumerate(pionE):
                    if piE < 0: # Flagging invalid pions in array (-8888)
                        transferred_mom[i] = -8888
                    else:
                        npions[0] += 1
                        piP     = pionP[i]
                        piTheta = pionTheta[i]
                    
                        pionP_T = piP * np.sin(piTheta)
                        pionP_L = piP * np.sin(piTheta)

                        transferred_mom[i] = (piE + muonE - pionP_L - muonP_L)**2 + (pionP_T + muonP_T)**2
            else:
                for i, piE in enumerate(pionE):
                    if piE > 0:
                        npions[0] += 1
                    transferred_mom[i] = -7777 # Flagging non matched events (-7777)
            "------------------------------------------------------"

            """ Second Event - Proton-Muon Transverse Momentum """
            "------------------------------------------------------"
            muonP_vec = ROOT.Math.XYZVector(muonPx, muonPy, muonPz)
            protonP_vec = ROOT.Math.XYZVector(protonPx, protonPy, protonPz)

            P_T, P_L = getLTMomenta(muonP_vec, protonP_vec)
            muon_proton_P_T[0] = P_T
            muon_proton_P_L[0] = P_L
            "------------------------------------------------------"

            """ Filling Second Event """
            "------------------------------------------------------"
            ana_tree.Fill()
            "------------------------------------------------------"
            n += 2
        
        else:
            track_time[0] = -9999
            beam_offset[0] = -9999
            for i in range(len(transferred_mom)):
                transferred_mom[i] = -9999
            muon_proton_P_T[0] = -9999
            muon_proton_P_L[0] = -9999
            npions[0] = -9999

            ana_tree.Fill()
            n += 1
        "------------------------------------------------------"
    "=========================================================="
    
    output_file.WriteObject(ana_tree, "Analysis")
    output_file.Close()

def CondenseAnalysis(output_filename):
    AnaChain = ROOT.TChain("Analysis", "AnaChain")
    
    for (root, dirs, files) in os.walk("./"):
        for file in files:
            if output_filename in file:
                AnaChain.Add(f"{root}/{file}")

    output_file = ROOT.TFile(f"{output_filename}_cond.root", 'recreate')
    
    output_tree = AnaChain.CloneTree(0)
    output_tree.CopyEntries(AnaChain)
    output_tree.SetName("Analysis")
    output_tree.SetTitle("Minerva Analysis")
    
    """ Analysis Histograms """
    "=========================================================="
    track_time_hstack = ROOT.THStack("TrackTimeHist", ";Track Time [ns]")
    track_time_hin = ROOT.TH1D("TrackTimeHist_in", ";Track Time [ns]", 10000, 0., 10000.)
    track_time_hout = ROOT.TH1D("TrackTimeHist_out", ";Track Time [ns]", 10000, 0., 10000.)
    track_time_hin.SetLineColor(1)
    track_time_hout.SetLineColor(2)

    beam_period = 18.831
    beam_size = beam_period/2 + 0.00001
    
    offset_hstack = ROOT.THStack("OffsetHist", ";Beam Offset [ns]")
    offset_hin = ROOT.TH1D( "OffsetHist_in", ";Offset From Peak [ns]", 250, -(beam_size + 0.5), (beam_size + 0.5) )
    offset_hout = ROOT.TH1D( "OffsetHist_out", ";Offset From Peak [ns]", 250, -(beam_size + 0.5), (beam_size + 0.5) )
    offset_htotal = ROOT.TH1D( "OffsetHist_Total", ";Offset From Peak [ns]", 250, -(beam_size + 0.5), (beam_size + 0.5) )
    offset_hin.SetLineColor(1)
    offset_hout.SetLineColor(2)
    
    print("Histograms Created")
    "=========================================================="

    """ Filling Histograms """
    "=========================================================="
    for event in AnaChain:
        beam_offset = event.beam_offset
        offset_htotal.Fill(beam_offset)
        if np.abs(beam_offset) > 5 and np.abs(beam_offset) < 100:
            track_time_hout.Fill(event.track_time)
            offset_hout.Fill(beam_offset)
        elif np.abs(beam_offset) <= 5:
            track_time_hin.Fill(event.track_time)
            offset_hin.Fill(beam_offset)
    "=========================================================="

    """ Histogram Drawing """
    "=========================================================="
    # Track Time
    track_time_hstack.Add(track_time_hout)
    track_time_hstack.Add(track_time_hin)
    "-----------------------------"
    
    """ Beam Offset """
    "-----------------------------"
    offset_hstack.Add(offset_hout)
    offset_hstack.Add(offset_hin)
    
    oCanvas = ROOT.TCanvas("BeamOffsetHist", "BeamOffsetHist")
    oCanvas.cd()
    offset_hstack.Draw()
    oCanvas.Update()
    
    # Defining function for a gaussian on top of a uniform background
    gaus_bg = ROOT.TF1("gaus_bg", "gaus(0) + [3]", -9.5,9.5) 
    gaus_bg.SetParName(3, "background")
    
    gaus_bg.SetParameters(2000, 0, 2, 20)

    # Fitting Packet Offset 
    hist_fit = offset_htotal.Fit("gaus_bg", "S Quiet", "", -9.5, 9.5)
    hist_fit.Draw()
    oCanvas.Update()
    "-----------------------------"

    output_file.WriteObject(output_tree, output_tree.GetName())
    output_file.WriteObject(track_time_hstack, "TrackTimeHist")
    output_file.WriteObject(oCanvas, "BeamOffsetHist")
    output_file.Close()

def ImportOffset(filename):
    file = open(f"{filename}.txt",'r')
    csvfile = csv.reader(file)
    for ttoffset in csvfile:
        return float(ttoffset[0])

def RunAnalysis(output_filename, file_set, set_numbers, tracktime_file):
    beam_start_time = ImportOffset(tracktime_file)
    for set_number in set_numbers:
        FileAnalysis(output_filename, file_set, set_number, beam_start_time)