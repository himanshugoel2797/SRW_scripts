
# %%
%matplotlib widget
# %%
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import numpy as np
from array import array
from srwlib import *
from srwlpy import *

# %%
def load_wfr_intens_pkl(filename, fdir):
    wfr = pickle.load(open(os.path.join(fdir, filename), 'rb'))
    mesh = wfr.mesh
    nx = wfr.mesh.nx
    ny = wfr.mesh.ny
    intens = array('f', nx*ny*[0])
    CalcIntFromElecField(intens, wfr, 6, 7, 3, 0.5 * (wfr.mesh.eStart + wfr.mesh.eFin), 0, 0)
    return np.array(intens).reshape((mesh.nx, mesh.ny)), mesh
def load_wfr_intens_h5(filename, fdir):
    intens, mesh, _, _ = srwl_uti_read_intens_hdf5(os.path.join(fdir, filename))
    intens = np.array(intens).reshape((mesh.ny, mesh.nx))
    return intens, mesh
def load_wfr_intens_dat(filename, fdir):
    intens, mesh = srwl_uti_read_intens_ascii(os.path.join(fdir, filename))
    intens = np.array(intens).reshape((mesh.ny, mesh.nx, mesh.ne))
    return intens, mesh

def load_wfr_intens(filename, fdir='', format=None):
    #Check the filename extension to figure out how to load the intensity data
    #.pkl = load_wfr_intens_pkl
    #.h5 = load_wfr_intens_h5
    #.dat = load_wfr_intens_dat
    if format is None:
        if filename.endswith('.pkl'):
            format = 'pkl'
        elif filename.endswith('.h5'):
            format = 'h5'
        elif filename.endswith('.dat'):
            format = 'dat'

    if format == 'pkl':
        return load_wfr_intens_pkl(filename, fdir)
    elif format == 'h5':
        return load_wfr_intens_h5(filename, fdir)
    elif format == 'dat':
        return load_wfr_intens_dat(filename, fdir)
    
    # If the filename does not match any of the expected formats, raise an error
    raise ValueError(f"Unsupported file format for {filename}")

#From https://joseph-long.com/writing/colorbars/
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def calc_fwhm(cut, xI, xF):
    x = np.linspace(xI, xF, len(cut))
    y = cut
    peak_idx = (np.abs(y-max(cut))).argmin()
    half_max = 0.5 * max(cut)

    left_idx = 0
    for i in range(peak_idx):
        if y[i] >= half_max:
            left_idx = i
            break

    right_idx = 0
    for i in range(len(y)-1, peak_idx, -1):
        if y[i] >= half_max:
            right_idx = i
            break
    
    return x[left_idx], x[right_idx]

def determine_lengthscale(v):
    if v < 1e-6:
        return 1e9, 'n'
    elif v < 1e-3:
        return 1e6, 'µ'
    elif v < 1:
        return 1e3, 'm'
    elif v < 1e3:
        return 1, ''
    elif v < 1e6:
        return 1e-3, 'k'
    elif v < 1e9:
        return 1e-6, 'M'
    else:
        return 1e-9, 'G'

def coord_r2px(mesh, x=None, y=None):
    if x is not None:
        xstep = (mesh.xFin - mesh.xStart) / mesh.nx
        x_px = int((x - mesh.xStart) / xstep)
    
    if y is not None:
        ystep = (mesh.yFin - mesh.yStart) / mesh.ny
        y_px = int((y - mesh.yStart) / ystep)
    
    if x is not None and y is not None:
        return x_px, y_px
    elif x is not None:
        return x_px
    elif y is not None:
        return y_px
    
def coord_px2r(mesh, x_px=None, y_px=None):
    if x_px is not None:
        xstep = (mesh.xFin - mesh.xStart) / mesh.nx
        x = mesh.xStart + x_px * xstep
    
    if y_px is not None:
        ystep = (mesh.yFin - mesh.yStart) / mesh.ny
        y = mesh.yStart + y_px * ystep
    
    if x_px is not None and y_px is not None:
        return x, y
    elif x_px is not None:
        return x
    elif y_px is not None:
        return y

def plot_intensity_with_profile(img, mesh, ax, show_fwhm=False, xa = 0.0, ya=0.0, axis_label = 'Intensity', unitStr= 'a.u.', logScale=False, logProfile=None):
    # Plot an intensity map with profile lines below and to the right of it
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div0 = make_axes_locatable(ax)
    ax_x = div0.append_axes("top", size="35%", pad=0.1, sharex=ax)
    ax_y = div0.append_axes("right", size="35%", pad=0.1, sharey=ax)

    # If logProfile is not specified, use the same setting as logScale
    if logProfile is None: 
        logProfile = logScale

    if logScale:
        img_log = np.log10(img, where=img>0)  # Avoid log(0) by setting a minimum value
        # Remove negative values for log scale

# no labels
    #ax_x.tick_params(axis="x", labelbottom=False)
    #ax_y.tick_params(axis="y", labelleft=False)

    lscale_fact, lscale_str = determine_lengthscale(((mesh.xFin - mesh.xStart) + (mesh.yFin - mesh.yStart))*0.5)
    lscale_str += 'm'

    print (img_log.max())
    # the scatter plot:
    im = ax.imshow(img_log if logScale else img, cmap='gray', extent=[(mesh.xStart) * lscale_fact, (mesh.xFin) * lscale_fact, -(mesh.yFin) * lscale_fact, -(mesh.yStart) * lscale_fact], vmin = 0 if logScale else None, vmax = img_log.max() if logScale else img.max())

    # extract vertical and horizontal cuts at the center of the image:
    x_px, y_px = coord_r2px(mesh, x=xa, y=ya)

    x = img[x_px, :]
    y = img[:, y_px][::-1]

    #Find FWHM of the vertical and horizontal cuts:
    if show_fwhm:
        xFWHM_i, xFWHM_f = calc_fwhm(x, mesh.xStart * lscale_fact, mesh.xFin * lscale_fact)
        yFWHM_i, yFWHM_f = calc_fwhm(y, mesh.yStart * lscale_fact, mesh.yFin * lscale_fact)

        ax_x.axvline(xFWHM_i, color='g', alpha=0.2, linestyle='--', label='FWHM')
        ax_x.axvline(xFWHM_f, color='g', alpha=0.2, linestyle='--')

        ax_y.axhline(yFWHM_i, color='g', alpha=0.2, linestyle='--')
        ax_y.axhline(yFWHM_f, color='g', alpha=0.2, linestyle='--')

        ax_x.set_title(f'FWHM: {xFWHM_f - xFWHM_i:.2f} {lscale_str}')
        ax_y.set_title(f'FWHM: {yFWHM_f - yFWHM_i:.2f} {lscale_str}')

    if logProfile:
        ax_x.semilogy(np.linspace(mesh.xStart * lscale_fact, mesh.xFin * lscale_fact, mesh.nx), x, color='r', linewidth=0.6)
        ax_y.semilogx(y, np.linspace(-mesh.yFin * lscale_fact, -mesh.yStart * lscale_fact, mesh.ny), color='r', linewidth=0.6)
    else:
        ax_x.plot(np.linspace(mesh.xStart * lscale_fact, mesh.xFin * lscale_fact, mesh.nx), x, color='r', linewidth=0.6)
        ax_y.plot(y, np.linspace(-mesh.yFin * lscale_fact, -mesh.yStart * lscale_fact, mesh.ny), color='r', linewidth=0.6)
    #ax_x.set_ylim(0, max(x) * 1.1)
    #ax_y.set_xlim(0, max(y) * 1.1)

    #ax_x.axes.xaxis.set_ticklabels([])
    #ax_y.axes.yaxis.set_ticklabels([])
    ax_x.tick_params(axis='x', labelbottom=False)
    ax_y.tick_params(axis='y', labelleft=False)

    ax_x.grid()
    ax_y.grid()
    ax_x.set_ylabel(axis_label + ' [' + unitStr + ']')
    ax_y.set_xlabel(axis_label + ' [' + unitStr + ']')

    if show_fwhm:
        ax_x.legend(loc = 'upper left')

    ax.set_xlabel(f'x [{lscale_str}]')
    ax.set_ylabel(f'y [{lscale_str}]')

    return ax, ax_x, ax_y

def plot_intensity(i, m, ax, showColorbar = True, logScale = False, unitStr = 'a.u.'):
    if logScale:
        i = np.log10(i, where=i>0)  # Avoid log(0) by setting a minimum value
        
    lscale_fact, lscale_str = determine_lengthscale((abs(m.xFin) + abs(m.xStart) + abs(m.yFin) + abs(m.yStart))*0.25)
    lscale_str += 'm'

    # Plot the intensity map
    pcm = ax.imshow(i, cmap='gray', extent=[(m.xStart) * lscale_fact, (m.xFin) * lscale_fact, -m.yFin * lscale_fact, -m.yStart * lscale_fact], vmin = 0 if logScale else i.min(), vmax = None if logScale else i.max())
    ax.set_xlabel(f'x [{lscale_str}]')
    ax.set_ylabel(f'y [{lscale_str}]')

    if showColorbar:
        cb = colorbar(pcm)
        cb.set_label('Intensity [' + unitStr + ']')

        if logScale:
            # Get the colorbar labels and format them to log scale
            cbar_ticks = cb.get_ticks()
            cbar_labels = [f'$10^{{{tick}}}$' for tick in cbar_ticks]
            cb.set_ticks(cbar_ticks)
            cb.set_ticklabels(cbar_labels)

    ax.set_aspect('equal')
    plt.tight_layout()

def plot_1d(i, m, ax, logScale = False, xlims = None, ylims = None, xlabel = 'Energy', ylabel = 'Intensity', xunit = 'eV', yunit = 'a.u.'):
    
    if m.nx > 1:
        aStart = m.xStart
        aFin = m.xFin
        lscale_fact, lscale_str = determine_lengthscale((abs(m.xFin) + abs(m.xStart)) * 0.5)
        lscale_str += 'm'
        data = i[:, m.ny // 2, m.ne // 2]
    elif m.ny > 1:
        aStart = m.yStart
        aFin = m.yFin
        lscale_fact, lscale_str = determine_lengthscale((abs(m.yFin) + abs(m.yStart)) * 0.5)
        lscale_str += 'm'
        data = i[m.nx // 2, :, m.ne // 2]
    elif m.ne > 1:
        aStart = m.eStart
        aFin = m.eFin
        lscale_fact, lscale_str = determine_lengthscale((abs(m.eFin) + abs(m.eStart)) * 0.5)
        data = i[m.nx // 2, m.ny // 2, :]
    else:
        raise ValueError('Invalid mesh')

    data_lscale_fact, data_lscale_str = determine_lengthscale((abs(data.max()) + abs(data.min())) * 0.5)

    pcm = ax.plot(np.linspace(aStart * lscale_fact, aFin * lscale_fact, len(data)), data * data_lscale_fact, color='r', linewidth=0.4)
    if logScale:
        ax.set_yscale('log')
    if xlims is not None:
        if len(xlims) == 2:
            ax.set_xlim(xlims[0], xlims[1])
        else:
            raise ValueError('Invalid xlims')
    if ylims is not None:
        if len(ylims) == 2:
            ax.set_ylim(ylims[0], ylims[1])
        else:
            raise ValueError('Invalid ylims')
    
    ax.grid('on', which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.5)
    ax.set_xlabel(f"{xlabel} [{lscale_str}{xunit}]")
    ax.set_ylabel(f"{ylabel} [{data_lscale_str}{yunit}]")

    plt.tight_layout()

def plot_cut_with_inset(i, m, ax, cut_pos=None, cut_orientation='v', inset_pos=[0.0, 0.51], inset_size=0.4, logScale=False, len_unit='m', mag_unit='a.u.', inset_2dmaps=True, imax=None, label=None, **kwargs):
    if cut_orientation == 'v':
        cut = i[:, cut_pos, m.ne // 2]
        cut_pos = coord_px2r(m, y_px=cut_pos)
        cut_unit = 'y'
        aStart, aFin = m.yStart, m.yFin
    elif cut_orientation == 'h':
        cut = i[cut_pos, :, m.ne // 2]
        cut_pos = coord_px2r(m, x_px=cut_pos)
        cut_unit = 'x'
        aStart, aFin = m.xStart, m.xFin
    else:
        raise ValueError('Invalid cut_orientation')

    lscale_fact, lscale_str = determine_lengthscale((abs(aFin) + abs(aStart)) * 0.5)
    lscale_str += len_unit

    ax.plot(np.linspace(aStart * lscale_fact, aFin * lscale_fact, len(cut)), cut, label=label, **kwargs)
    if logScale:
        ax.set_yscale('log')
    ax.grid('on', which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.5)
    if cut_orientation == 'v':
        ax.set_xlabel(f"Vertical Position [{lscale_str}]")
    elif cut_orientation == 'h':
        ax.set_xlabel(f"Horizontal Position [{lscale_str}]")
    ax.set_ylabel(f"Intensity [{mag_unit}]")

    if imax is not None:
        ax.set_ylim(None, imax)

    # Add inset
    if inset_2dmaps:
        inset_ax = ax.inset_axes([inset_pos[0], inset_pos[1], inset_size, inset_size])
        inset_ax.imshow(i, cmap='gray', extent=[(m.xStart) * lscale_fact, (m.xFin) * lscale_fact, -m.yFin * lscale_fact, -m.yStart * lscale_fact], vmax = imax)
        
        #hide tick labels
        inset_ax.set_xticklabels([])
        inset_ax.set_yticklabels([])
        return ax, inset_ax
    return ax, None

def plot_cms(cms, cm_idxs, orientation = 'v', inset_2dmaps = True, mag_unit='a.u.', logScale=False, last_cum_intens=False):
    if not inset_2dmaps:
        raise NotImplementedError('Only inset 2D maps are supported at the moment')
    #if orientation != 'v':
    #    raise NotImplementedError('Only vertical orientation is supported at the moment')
    
    #Load the CMs
    max_cm_idx = max(cm_idxs) + 1
    cms = cms[:max_cm_idx]
    cm_intens = []
    cm_cuts = []
    cm_meshes = []

    if orientation == 'v':
        fig, axs = plt.subplots(len(cm_idxs), 1, figsize=(4.5, len(cm_idxs) * 4))
        inset_pos = [0.0, 0.51]
    elif orientation == 'h':
        fig, axs = plt.subplots(1, len(cm_idxs), figsize=(len(cm_idxs) * 4, 4.))
        inset_pos = [0.02, 0.51]

    #Extract intensity for each CM
    cm_idx = 0
    init_max = None
    for cm in cms:
        #Extract the horizontal cut at the center of the image for each CM
        mesh = cm.mesh
        nx =   cm.mesh.nx
        ny =   cm.mesh.ny
        intens = array('f', nx*ny*[0])
        CalcIntFromElecField(intens, cm, 6, 7, 3, 0.5 * (mesh.eStart + mesh.eFin), 0, 0)

        cm_arr = np.array(intens).reshape((mesh.nx, mesh.ny, 1))

        cm_intens.append(cm_arr)
        cm_meshes.append(mesh)
        sum_intens = np.sum(cm_intens, axis=0)
        init_max = sum_intens.max() * 1.1
        cm_idx += 1

    cm_idx = 0
    for cm in cms:
        #Plot the accumulating intensity cuts for each CM with the corresponding 2D map as an inset
        if cm_idx in cm_idxs:
            plt_idx = cm_idxs.index(cm_idx)
            ax = axs[plt_idx]
            cm_arr = cm_intens[cm_idx]
            mesh = cm_meshes[cm_idx]
            sum_intens = np.sum(cm_intens[:cm_idx + 1], axis=0)
            
            if cm_idx == cm_idxs[-1] and last_cum_intens:
                ax.set_title(f'Accumulated Intensity ({cm_idx + 1} CMs)')
                plot_intensity(sum_intens, mesh, ax, logScale=logScale, unitStr=mag_unit, showColorbar=False)
            else:
                for i in range(cm_idx + 1):
                    ax, _ = plot_cut_with_inset(cm_intens[i], cm_meshes[i], ax, cut_pos=cm_meshes[i].ny // 2, cut_orientation='h', logScale=logScale, inset_2dmaps=False, len_unit='m', mag_unit=mag_unit, imax=init_max, label=f'CM #{i}', linestyle='-.', linewidth = 0.7)
                ax, ax_inset = plot_cut_with_inset(sum_intens, mesh, ax, cut_pos=mesh.ny // 2, cut_orientation='h', inset_pos=inset_pos, inset_size=0.4, logScale=logScale, len_unit='m', mag_unit=mag_unit, imax=init_max, label='Sum', linewidth = 0.7, color='r')
                ax_inset.set_title(f'Sum')
                ax.legend()
        
        cm_idx += 1


    plt.tight_layout()
    return fig, axs

if __name__ == "__main__":
    # Example usage
    #filename = 'wfr_3d_0.pkl'  # Replace with your actual filename
    filename = "intensity_0.h5"
    #filename = 'init_res_ec_spec_en.dat'
    #fdir = 'cxfel-0/probes'  # Replace with your actual directory
    #fdir = 'xpp-multipulse-resmeas-allJit-1.0specJit-224.5-JFFixedOffcen-20250224-1/at_detector_merged_s200_c400_u8-1'
    #fdir = 'xpp-multipulse-resmeas-allJit-1.0specJit-224.5-JFFixedOffcen-overlap70pc-202500305-1'
    #fdir = 'apsu-lowsampleposjit-EIGER-60ms-20250224-1/at_detector_merged_s200_c400_u8-1'
    fdir = 'apsu-lowsampleposjit-EIGER-20250428-0/at_detector_merged_s200_c400_u8-0'
    #fdir = 'apsu-atSmp-1/at_detector_merged_s200_c400_u8-1'
    #fdir = 'cxfel-0/at_detector_merged_s200_c400_u8-1'
    #fdir = 'cxfel-0'
    #filename = 'init_dcm_ec_fluence_xy.dat'

    i, m = load_wfr_intens(filename, fdir, format='h5')
    print ("{:.2e}".format(np.sum(i) * (75e-06 * 1000)**2 * 1e-3))
    print ("{:.2e}".format(np.sum(i)))
    print (np.sum(i) * (75e-06 * 1000)**2 * 1e-3 * 0.06)
    print (np.max(i))

    #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    #plot_intensity_with_profile(i, m, ax, show_fwhm=False, logScale=True, axis_label="Photon Count", unitStr="$ph$")#, logProfile=False)
    #plot_intensity_with_profile(i, m, ax, show_fwhm=False, logScale=True, axis_label="Fluence", unitStr="$J/mm^2$")#, logProfile=False)
    #plot_intensity(i, m, ax, logScale=True, showColorbar=False)
    

    wfr = srwl_uti_read_wfr_cm_hdf5(_file_path = os.path.join('apsu', 'aps_u_33id_bef_bl_cm.h5'))
    print (wfr[0].mesh.eStart)
    print (wfr[0].mesh.xFin - wfr[0].mesh.xStart)
    #plot_cms(wfr, [0, 4], last_cum_intens=False, mag_unit='$ph/s/0.1\%bw/mm^2$', orientation='v')

    #i, m = load_wfr_intens(filename, fdir)    
    #fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    #plot_1d(i, m, ax, xlabel='Energy', ylabel='Spectral Energy', xunit='eV', yunit='J/eV', logScale=False)
#
    #i0, m0 = load_wfr_intens('init_res_ec_fluence_xy.dat', fdir)
    #inset_ax = ax.inset_axes([0.6, 0.35, 0.5, 0.5])
    #plot_intensity(i0, m0, inset_ax, logScale=False, unitStr='J/mm^2', showColorbar=False)
    #inset_ax.set_title('Fluence')
    #inset_ax.set_xlim(-100, 100)
    #inset_ax.set_ylim(-100, 100)
#
    #i0, m0 = load_wfr_intens('init_res_ec_spec_fluence_xy.dat', fdir)
    #inset_ax = ax.inset_axes([-0.1, 0.35, 0.5, 0.5])
    #plot_intensity(i0, m0, inset_ax, logScale=False, unitStr='J/mm^2', showColorbar=False)
    #inset_ax.set_title('Spectral Fluence\nat 8800.046 eV')
    #inset_ax.set_xlim(-100, 100)
    #inset_ax.set_ylim(-100, 100)
    #inset_ax.set_xlabel("")
    #inset_ax.set_ylabel("")
    #inset_ax.set_xticklabels([])
    #inset_ax.set_yticklabels([])


    plt.show()
# %%

