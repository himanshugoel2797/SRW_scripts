import xraylib
import numpy as np

mat = "B"
dens = xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(mat)) #g/cm^3
energy_keV = 9 #keV

refr_idx_dec = 1 - xraylib.Refractive_Index_Re(mat,energy_keV,dens)
refr_idx_im = xraylib.Refractive_Index_Im(mat,energy_keV,dens)
abs_len = 0.01/(xraylib.CS_Total_CP(mat, energy_keV)*dens)  #NOTE: https://github.com/tschoonj/xraylib/issues/42#issuecomment-272913467 # CS_Total_CP gives mass attenuation coefficient in cm^2/g, dens is in g/cm^3, so abs_len is in cm^-1, convert to m^-1

lambda_ = 12.39842/energy_keV * 1e-10  # in meters

print("Refractive Index Decrement: " + str(refr_idx_dec))
print("Attenuation Length from imaginary component of refractive index: " + str((4.135e-16 * 2999792458)/(4*np.pi*energy_keV*1000*refr_idx_im)))
print("Attenuation Length from mass attenuation coefficient: " + str(abs_len))

