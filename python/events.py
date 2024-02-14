from dataclasses import dataclass

import numpy as np

from astropy import cosmology as cosmo
from astropy import units as u

from scipy.interpolate import interp1d

seconds_in_year = 525600*60

def Mceta_from_m1m2(m1, m2):
    # Function to compute the chirp mass and symmetric mass ratio of a binary given its component masses
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    eta = (m1*m2)/((m1+m2)*(m1+m2))
    return Mc, eta

def redshift_Mc(Mc, z):
    # Function to compute the detector frame chirp mass from a chirp mass at a redshift z
    return Mc*(1+z)

def md_sfr_td(z, alpha=1.801, beta=3.492, zp=1.8334):
    """Madau & Dickinson-like function refit after integration with SFR and
    the time delay between binary formation and merger with a minimum of 20Myr
    and a maximum of 10 Gyr."""
    return (1+z)**alpha / (1 + ((1+z) / (1 + zp))**(alpha + beta))

def md_sfr(z, alpha=2.7, beta=2.9, zp=1.9):
    """SFR from Madau & Dickinson 2014 with parameters"""
    return (1+z)**alpha / (1 + ((1+z) / (1 + zp))**(alpha + beta))

def redshift_volume(zi):
    """Cosmology stuff"""
    dVdz = 4.0 * np.pi * cosmo.Planck18.differential_comoving_volume(zi).to(u.Gpc**3/u.sr).value
    V_of_z = interp1d(zi, dVdz)
    return V_of_z

@dataclass
class Event:
    # redshift: float  # z
    luminosity_distance: float  # D_L
    redshift: float  # z
    chirp_mass: float  # M_c
    symmetric_mass_ratio: float  # eta
    coalescence_time: float  # t_c
    coalescence_phase: float  # phi_c
    polar_angle: float  # theta
    azimuthal_angle: float  # phi
    inclination: float  # iota
    polarization_angle: float  # psi
    spin1z: float  # chi_1z
    spin2z: float  # chi_2z
    tidal_deformability1: float  # Lambda_1
    tidal_deformability2: float  # Lambda_2

def compute_lambda_tilde(Lambda1: float, Lambda2: float, eta: float) -> np.ndarray:
    Delta = np.sqrt(1 - 4 * eta)
    Lambda_t = (8/13) * ((1 + 7*eta - 31*eta**2) * (Lambda1 + Lambda2) + Delta * (1 + 9*eta - 11*eta**2) * (Lambda1 - Lambda2))
    DeltaLambda_t = 0.5 * ((Delta * (1319 - 13272*eta + 8944*eta**2)) / (1319) * (Lambda1 + Lambda2) + (1319 - 15910*eta + 32850*eta**2 + 3380*eta**3) / (1319) * (Lambda1 - Lambda2))
    return Lambda_t, DeltaLambda_t

def make_events(num_of_days, BNSRate=105.5, z_min=0, z_max=20, time_delay=True):
    ## interpolate m(lambda)
    m_lambda = np.loadtxt('./eos_files/m_lambda.txt')
    m_lambda_interp = interp1d(m_lambda[:, 0], m_lambda[:, 1], kind='cubic', bounds_error=False, fill_value=np.nan)
    frac_of_year = num_of_days / 365.25
    zi = np.linspace(z_min, z_max, int(10000))
    volume_of_z = redshift_volume(zi)
    if time_delay:
        volume = md_sfr_td(zi) * volume_of_z(zi) / (1+zi) * BNSRate / md_sfr_td(0)
    else:
        volume = md_sfr(zi) * volume_of_z(zi) / (1+zi) * BNSRate / md_sfr(0)
    # plt.plot(zi, volume)
    nevents = int(np.trapz(volume, zi) * frac_of_year)
    print('Making {} events'.format(nevents))
    z = np.random.choice(zi, size=nevents, p=volume / np.sum(volume))

    # intrinsic parameters
    dLs = cosmo.Planck18.luminosity_distance(z).to(u.Gpc).value
    m1 = np.random.uniform(1, 2, nevents)
    m2 = np.random.uniform(1, 2, nevents)
    Mcs, etas = Mceta_from_m1m2(m1, m2)
    Mcs = redshift_Mc(Mcs, z)
    tc = np.sort(np.random.uniform(0, seconds_in_year * frac_of_year, nevents))
    phic = np.random.uniform(0, 2*np.pi, nevents)
    chi1z = np.random.uniform(-0.05, 0.05, nevents)
    chi2z = np.random.uniform(-0.05, 0.05, nevents)
    Lambda1 = m_lambda_interp(m1)
    Lambda2 = m_lambda_interp(m2)

    LambdaTilde, deltaLambda = compute_lambda_tilde(Lambda1, Lambda2, etas)

    # extrinsic parameters
    theta = np.arccos(np.random.uniform(-1, 1, nevents))
    phi = np.random.uniform(0, 2*np.pi, nevents)
    iota = np.arccos(np.random.uniform(-1, 1, nevents))
    psi = np.random.uniform(0, np.pi, nevents)

    # events = {'Mc':np.array(Mcs), 'eta':np.array(etas), 'tGPS':np.array(tc), 'Phicoal':np.array(phic),
    #           'theta':np.array(theta), 'chi1z':np.array(chi1z), 'chi2z':np.array(chi2z),
    #           'Lambda1':np.array(Lambda1), 'Lambda2':np.array(Lambda2), 'LambdaTilde':np.array(LambdaTilde),
    #           'deltaLambda':np.array(deltaLambda), 'z':np.array(z),
    #           'dL':np.array(dLs), 'phi':np.array(phi), 'iota':np.array(iota), 'psi':np.array(psi)}

    events = [Event(luminosity_distance=dLs[i], redshift=z[i], chirp_mass=Mcs[i],
                    symmetric_mass_ratio=etas[i], coalescence_time=tc[i],
                    coalescence_phase=phic[i], polar_angle=theta[i],
                    azimuthal_angle=phi[i], inclination=iota[i],
                    polarization_angle=psi[i], spin1z=chi1z[i], spin2z=chi2z[i],
                    tidal_deformability1=Lambda1[i],
                    tidal_deformability2=Lambda2[i]) for i in range(nevents)]

    return events
