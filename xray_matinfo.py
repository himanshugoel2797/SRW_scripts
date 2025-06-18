import xraylib
import numpy as np

mat = "Cu"
dens = xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(mat)) #g/cm^3
energy_keV = 8.8 #keV

refr_idx_dec = 1 - xraylib.Refractive_Index_Re(mat,energy_keV,dens)
refr_idx_im = xraylib.Refractive_Index_Im(mat,energy_keV,dens)
abs_len = 0.01/(xraylib.CS_Energy_CP(mat, energy_keV)*dens)  #NOTE: https://github.com/tschoonj/xraylib/issues/42#issuecomment-272913467 # CS_Energy_CP gives mass attenuation coefficient in cm^2/g, dens is in g/cm^3, so abs_len is in cm, convert to m

lambda_ = 12.39842/energy_keV * 1e-10  # in meters

print("Refractive Index Decrement: " + str(refr_idx_dec))
print("Attenuation Length from imaginary component of refractive index: " + str(lambda_/(4*np.pi*refr_idx_im)))
print("Attenuation Length from mass attenuation coefficient: " + str(abs_len))

