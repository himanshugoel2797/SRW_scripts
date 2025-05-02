import xraylib

mat = "Cu"
dens = xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(mat))
energy_keV = 8.8

refr_idx = 1 - xraylib.Refractive_Index_Re(mat,energy_keV,dens)
abs_len = 0.01/(xraylib.CS_Energy_CP(mat, energy_keV)*dens)

print("Refractive Index: " + str(refr_idx))
print("Attenuation Length: " + str(abs_len))

