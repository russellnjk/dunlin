import numpy as np

def model_coarse_1(t, y, params, inputs):

	x = y[0]
	S = y[1]
	P = y[2]
	Q = y[3]
	R = y[4]
	M = y[5]
	H = y[6]

	Vrctr     = params[0]
	Mcell     = params[1]
	Dcell     = params[2]
	v_uptake  = params[3]
	k_uptake  = params[4]
	yield_S   = params[5]
	v_synprot = params[6]
	k_synprot = params[7]
	k_fr      = params[8]
	n_fr      = params[9]
	fq        = params[10]
	fr        = params[11]
	fm        = params[12]
	fh        = params[13]
	k_ind     = params[14]

	dil = inputs[0]
	Sin = inputs[1]
	ind = inputs[2]

	
	all_prot = Q +R +M +H
	Vcell    = all_prot/Dcell
	
	uptake       = v_uptake*x*S/(S + k_uptake)
	total_uptake = uptake/6e23*1e6/Vrctr
	
	P_conc  = (P/6e23)/Vcell
	r_sat   = P_conc/(P_conc + k_synprot)
	synprot = v_synprot*R*r_sat
	
	PR  = P/R
	fr_ = fr*PR**n_fr/(k_fr**n_fr + PR**n_fr)
	
	fh_ = fh*ind/(k_ind + ind)
	
	synQ = synprot*fq
	synM = synprot*(1-fq)*fm /(fm + fr_ + fh)
	synR = synprot*(1-fq)*fr_/(fm + fr_ + fh)
	synH = synprot*(1-fq)*fh /(fm + fr_ + fh)
	
	mu = (synQ +synM +synR)/Mcell
	
	dS = -dil*S +dil*Sin -total_uptake
	dx = -dil*x +x*mu
	dP = -mu*P +uptake*yield_S -synprot
	dQ = -mu*Q +synQ/300
	dR = -mu*R +synR/7459
	dM = -mu*M +synM/300
	dH = -mu*H +synH/300

	return np.array([dx, dS, dP, dQ, dR, dM, dH])