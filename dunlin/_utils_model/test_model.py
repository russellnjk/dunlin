import numpy as np
from numba import njit

#Cell density (protein conc)in uM
#Cell volume in L
	
#@njit
def model_coarse_1(t, y, p, rm):
	#States
	x = y[0]
	S = y[1]
	P = y[2]
	Q = y[3]
	R = y[4]
	M = y[5]
	H = y[6]

	#Params
	Mcell     = p[0]
	v_uptake  = p[1]
	k_uptake  = p[2]
	n_uptake  = p[3]
	ntf       = p[4]
	yield_S   = p[5]
	v_synprot = p[6]
	k_synprot = p[7]
	k_fr      = p[8]
	n_fr      = p[9]
	fq        = p[10]
	fr        = p[11]
	fm        = p[12]
	fh        = p[13]
	k_ind     = p[14]
	k_cm      = p[15]
	dil       = p[16]
	Sin       = p[17]
	ind       = p[18]
	Cm        = p[19]
	
	#Equations
	#Variables
	n_H          = H/300
	n_Q          = Q/300
	n_M          = M/300
	n_R          = R/7459
	cells_per_OD = 2.4e9
	uM_cells     = x*cells_per_OD/6e23*1e3*1e6
	uptake       = v_uptake*n_M*S**n_uptake/(S**n_uptake + k_uptake**n_uptake)
	total_uptake = uptake*uM_cells
	all_prot     = Q +R +M +H
	P_conc       = P/all_prot
	r_sat        = P_conc/(P_conc + k_synprot)
	r_inh        = k_cm/(Cm + k_cm)
	synprot      = v_synprot*n_R*r_sat*r_inh
	PR           = P/R
	reg          = PR**n_fr/(k_fr**n_fr + PR**n_fr) *(1-0.046) + 0.046
	fr_          = fr*reg
	fh_          = fh*ind/(k_ind + ind)
	mu           = synprot/Mcell
	#mu           = (synQ +synM +synR)/Mcell
	
	#Reactions
	dil_S = dil*S
	dil_x = dil*x
	mu_x  = mu*x
	mu_P  = mu*P
	mu_Q  = mu*Q
	mu_R  = mu*R
	mu_M  = mu*M
	mu_H  = mu*H
	in_S  = dil*Sin
	up_S  = total_uptake
	synP  = uptake*yield_S
	synQ  = synprot*fq
	synM  = synprot*(1-fq)*fm /(fm + fr_ + fh_)
	synR  = synprot*(1-fq)*fr_/(fm + fr_ + fh_)
	synH  = synprot*(1-fq)*fh_/(fm + fr_ + fh_)
	con_P = synprot
	
	#Differentials
	d_x = +rm[0, 0]*dil_S +rm[1, 0]*dil_x +rm[2, 0]*mu_x +rm[3, 0]*mu_P +rm[4, 0]*mu_Q +rm[5, 0]*mu_R +rm[6, 0]*mu_M +rm[7, 0]*mu_H +rm[8, 0]*in_S +rm[9, 0]*up_S +rm[10, 0]*synP +rm[11, 0]*synQ +rm[12, 0]*synR +rm[13, 0]*synM +rm[14, 0]*synH +rm[15, 0]*con_P
	d_S = +rm[0, 1]*dil_S +rm[1, 1]*dil_x +rm[2, 1]*mu_x +rm[3, 1]*mu_P +rm[4, 1]*mu_Q +rm[5, 1]*mu_R +rm[6, 1]*mu_M +rm[7, 1]*mu_H +rm[8, 1]*in_S +rm[9, 1]*up_S +rm[10, 1]*synP +rm[11, 1]*synQ +rm[12, 1]*synR +rm[13, 1]*synM +rm[14, 1]*synH +rm[15, 1]*con_P
	d_P = +rm[0, 2]*dil_S +rm[1, 2]*dil_x +rm[2, 2]*mu_x +rm[3, 2]*mu_P +rm[4, 2]*mu_Q +rm[5, 2]*mu_R +rm[6, 2]*mu_M +rm[7, 2]*mu_H +rm[8, 2]*in_S +rm[9, 2]*up_S +rm[10, 2]*synP +rm[11, 2]*synQ +rm[12, 2]*synR +rm[13, 2]*synM +rm[14, 2]*synH +rm[15, 2]*con_P
	d_Q = +rm[0, 3]*dil_S +rm[1, 3]*dil_x +rm[2, 3]*mu_x +rm[3, 3]*mu_P +rm[4, 3]*mu_Q +rm[5, 3]*mu_R +rm[6, 3]*mu_M +rm[7, 3]*mu_H +rm[8, 3]*in_S +rm[9, 3]*up_S +rm[10, 3]*synP +rm[11, 3]*synQ +rm[12, 3]*synR +rm[13, 3]*synM +rm[14, 3]*synH +rm[15, 3]*con_P
	d_R = +rm[0, 4]*dil_S +rm[1, 4]*dil_x +rm[2, 4]*mu_x +rm[3, 4]*mu_P +rm[4, 4]*mu_Q +rm[5, 4]*mu_R +rm[6, 4]*mu_M +rm[7, 4]*mu_H +rm[8, 4]*in_S +rm[9, 4]*up_S +rm[10, 4]*synP +rm[11, 4]*synQ +rm[12, 4]*synR +rm[13, 4]*synM +rm[14, 4]*synH +rm[15, 4]*con_P
	d_M = +rm[0, 5]*dil_S +rm[1, 5]*dil_x +rm[2, 5]*mu_x +rm[3, 5]*mu_P +rm[4, 5]*mu_Q +rm[5, 5]*mu_R +rm[6, 5]*mu_M +rm[7, 5]*mu_H +rm[8, 5]*in_S +rm[9, 5]*up_S +rm[10, 5]*synP +rm[11, 5]*synQ +rm[12, 5]*synR +rm[13, 5]*synM +rm[14, 5]*synH +rm[15, 5]*con_P
	d_H = +rm[0, 6]*dil_S +rm[1, 6]*dil_x +rm[2, 6]*mu_x +rm[3, 6]*mu_P +rm[4, 6]*mu_Q +rm[5, 6]*mu_R +rm[6, 6]*mu_M +rm[7, 6]*mu_H +rm[8, 6]*in_S +rm[9, 6]*up_S +rm[10, 6]*synP +rm[11, 6]*synQ +rm[12, 6]*synR +rm[13, 6]*synM +rm[14, 6]*synH +rm[15, 6]*con_P
	
	# d_S = -dil*S +dil*Sin -total_uptake
	# d_x = -dil*x +x*mu
	# d_P = -mu*P +synP -synprot
	# d_Q = -mu*Q +synQ
	# d_R = -mu*R +synR
	# d_M = -mu*M +synM
	# d_H = -mu*H +synH
	
	return np.array([d_x, d_S, d_P, d_Q, d_R, d_M, d_H])
