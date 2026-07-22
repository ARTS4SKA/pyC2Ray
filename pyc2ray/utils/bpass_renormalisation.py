import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst
from astropy.modeling.physical_models import BlackBody
from scipy.integrate import quad, quad_vec

# TO DO: Move Physical constants (ionisation edges) to parameter file
E_HI   = 13.598 * u.eV
E_HeI  = 24.587 * u.eV
E_HeII = 54.416 * u.eV
 
FREQ_HI   = (E_HI   / cst.h).to(u.Hz)
FREQ_HeI  = (E_HeI  / cst.h).to(u.Hz)
FREQ_HeII = (E_HeII / cst.h).to(u.Hz)
 
LAMB_HI   = (cst.c / FREQ_HI  ).to(u.AA)
LAMB_HeI  = (cst.c / FREQ_HeI ).to(u.AA)
LAMB_HeII = (cst.c / FREQ_HeII).to(u.AA)

""" __________________________ FILE LOADING UTILITIES __________________________ """
def load_starmass(path):
    """
    Load surviving stellar mass file
    """
    return pd.read_csv(path, sep=r'\s+', engine='python', names=['log_age', 'surv', 'remnant'])

def load_ionizing_flux(path):
    """
    Load One ionizing flux file
    """
    return pd.read_csv(path, sep=r'\s+', engine='python',
                       names=['log_age', 'prod_rate', 'halpha', 'FUV', 'NUV'])

def load_yields(path):
    """
    Load One yelds file
    """
    return pd.read_csv(path, sep=r'\s+', engine='python',
                       names=['log_age', 'H_sw', 'He_sw', 'Z_sw', 'E_sw', 'E_sn', 'H_sn', 'He_sn', 'Z_sn'])

def load_sed(path):
    """
    Load One SED file
    """
    df = pd.read_csv(path, sep=r"\s+", engine='python', header=None)
    if df.columns.shape[0] == 26:
        # this to take into account the case where I'm using the files with half time res
        cols = ['WL', '6.0', '6.2', '6.4', '6.6', '6.8', '7.0', '7.2', '7.4', '7.6',
                '7.8', '8.0', '8.2', '8.4', '8.6', '8.8', '9.0', '9.2', '9.4', '9.6',
                '9.8', '10.0', '10.2', '10.4', '10.6', '10.8']
        
    if df.columns.shape[0] == 52:
        cols = ['WL', '6.0', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8',
                '6.9', '7.0', '7.1', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '7.8',
                '7.9', '8.0', '8.1', '8.2', '8.3', '8.4', '8.5', '8.6', '8.7', '8.8',
                '8.9', '9.0', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.7', '9.8',
                '9.9', '10.0', '10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7',
                '10.8', '10.9', '11.0']
    df.columns = cols
    return df

def sed_to_lum_freq(sed, age_key):
    """
    Extract a single age column from a loaded SED and convert it to a
    monochromatic luminosity per unit frequency.
 
    Parameters
    ----------
    sed : pandas.DataFrame
        Output of `load_sed`.
    age_key : str
        Column name selecting log10(age/yr), e.g. '7.0'.
 
    Returns
    -------
    freq : Quantity [Hz]
    lum_freq : Quantity [Lsun / Hz]
    """
    lamb = np.asarray(sed['WL']) * u.AA
    lum_lamb = np.asarray(sed[age_key]) * (u.Lsun / u.AA)
 
    freq = (cst.c / lamb).to(u.Hz)
    # L_nu = L_lambda * lambda^2 / c
    lum_freq = (lum_lamb * lamb**2 / cst.c).to(u.Lsun / u.Hz)
 
    return freq, lum_freq

def metallicity_from_filename(name):
    """
    Extract the Z value from a BPASS filename.

    BPASS encodes metallicity as a zero-padded fixed-point string after
    'z', with the leading '0.' stripped. For example:

        'spectra-bin-imf135_300.z00001.dat' -> 0.00001
        'spectra-bin-imf135_300.z020.dat'   -> 0.020

    Parameters
    ----------
    name : str
        Filename (not full path).

    Returns
    -------
    float
        Metallicity Z.
    """
    m = re.search(r'\.z(\d+)\.', name)
    if m is None:
        raise ValueError(f"Could not parse metallicity from {name!r}")
    return float('0.' + m.group(1))

""" __________________________ BLACK BODY UTILITIES __________________________ """
def planck_spectrum_nu(nu, T=1e5 * u.K):
    """Planck specific intensity B_nu(T). Returns W / (m^2 Hz sr)."""
    if not isinstance(nu, u.Quantity) or not nu.unit.is_equivalent(u.Hz):
        raise ValueError("nu must be an astropy Quantity with frequency units.")
 
    bb = BlackBody(temperature=T.to(u.K))
    return bb(nu.to(u.Hz)).to(u.W / (u.m**2 * u.Hz * u.sr))
 
 
def monochromatic_luminosity(nu, T=1e5 * u.K, R=cst.R_sun):
    """
    Black body monochromatic luminosity for a sphere of radius R at
    temperature T:
 
        L_nu = 4 pi R^2 * pi * B_nu(T)
 
    Returns Quantity [Lsun / Hz].
    """
    B_nu = planck_spectrum_nu(nu, T=T)
    F_nu = (np.pi * u.sr) * B_nu                 # emergent flux at surface
    L_nu = 4 * np.pi * R**2 * F_nu
    return L_nu.to(u.Lsun / u.Hz)
 
 
def total_luminosity(T=1e5 * u.K, R=cst.R_sun):
    """Stefan-Boltzmann bolometric luminosity of a sphere (Lsun)."""
    return (4 * np.pi * R**2 * cst.sigma_sb * T**4).to(u.Lsun)

""" __________________________ SPECTRUM INTERPOLATION __________________________ """
def lum_freq_at_nu(target_nu, freq, lum_freq):
    """
    Log-log interpolate a monochromatic luminosity onto a single target
    frequency.
 
    Parameters
    ----------
    target_nu : Quantity [Hz]
    freq : Quantity [Hz]
    lum_freq : Quantity
        Luminosity per unit frequency on `freq`.
 
    Returns
    -------
    Quantity with the same units as `lum_freq`.
    """
    # Unit safety:
    target_nu = target_nu.to(u.Hz)
    freq = freq.to(u.Hz)
    unit = lum_freq.unit

    # Convert to numpy arrays:
    nu_vals = freq.value
    L_vals = lum_freq.value

    # Sort:
    order = np.argsort(nu_vals)
    nu_vals = nu_vals[order]
    L_vals = L_vals[order]

    # Safety checks:
    if np.any(nu_vals <= 0):
        raise ValueError("Frequency grid contains non-positive values.")
    if np.any(L_vals < 0):
        raise ValueError("Luminosity contains negative values.")

    # Bounds check:
    if not (nu_vals[0] <= target_nu.value <= nu_vals[-1]):
        raise ValueError(
            f"target_nu={target_nu} lies outside the interpolation range "
            f"[{nu_vals[0]:.3e} Hz, {nu_vals[-1]:.3e} Hz]"
        )

    # If surrounding values were truly zero, return exact zero
    if np.all(L_vals == 0):
        return 0 * unit

    # Replace zeros with tiny floor for log interpolation:
    tiny = np.finfo(float).tiny
    L_safe = np.where(L_vals == 0, tiny, L_vals)

    # Log-log interpolation
    log_L = np.interp(
        np.log10(target_nu.value),
        np.log10(nu_vals),
        np.log10(L_safe),
    )
 
    return (10**log_L) * unit
    
""" __________________________ SPECTRUM NORMLAISATION __________________________ """
def normalise_luminosity_spectrum(lum_freq,
                                  freq,
                                  T=1e5 * u.K,
                                  R=cst.R_sun):
    """
    Rescale a spectrum so that its bolometric integral over `freq`
    equals that of a (T, R) black body on the same grid:
 
        L_nu_norm(nu) = L_nu(nu) * [ Int L_nu_BB dnu / Int L_nu dnu ]
 
    Both integrands are sampled on `freq`, so this is a fair comparison
    only over the wavelength range that BPASS itself covers.
    """
    bb_lum_freq = monochromatic_luminosity(freq, T=T, R=R)
 
    # Frequencies in the array are typically descending (because the
    # underlying wavelength grid is ascending), so the integral picks up
    # a sign which we flip with -1.
    int_bb = -np.trapezoid(bb_lum_freq, freq)
    int_data = -np.trapezoid(lum_freq, freq)
 
    scaling = (int_bb / int_data).decompose().value
    return (lum_freq * scaling).to(u.Lsun / u.Hz)

""" __________________________ ANCHOR VALUE EXTRACTION __________________________ """
def normalised_anchor_value(sed, age_key,
                            T=1e5 * u.K, R=cst.R_sun,
                            anchor_freq=FREQ_HI):
    """
    For one BPASS SED column at one age, return the normalised
    monochromatic luminosity at the anchor frequency.

    "Normalised" = bolometrically rescaled so that the BPASS spectrum's
    integral over the BPASS frequency grid equals that of the reference
    (T, R) black body on the same grid.

    Parameters
    ----------
    sed : pandas.DataFrame
        Output of `load_sed`.
    age_key : str
        Age column name, e.g. '7.0'.
    T, R : Quantity
        Reference black body parameters used by the normalisation.
    anchor_freq : Quantity [Hz]
        Frequency at which to evaluate the normalised spectrum.

    Returns
    -------
    float
        Normalised L_nu at `anchor_freq`, in Lsun/Hz (plain float,
        units stripped — the value is what enters the ratio).
    """
    freq, lum_freq = sed_to_lum_freq(sed, age_key)
    norm_bpass = normalise_luminosity_spectrum(lum_freq, freq, T=T, R=R)
    val = lum_freq_at_nu(anchor_freq, freq, norm_bpass)
    return val.to(u.Lsun / u.Hz).value


def anchor_values_for_file(sed, max_log_age=7.0,
                           T=1e5 * u.K, R=cst.R_sun,
                           anchor_freq=FREQ_HI):
    """
    Compute the normalised anchor value for every retained age column
    in one BPASS file.

    Parameters
    ----------
    sed : pandas.DataFrame
        Output of `load_sed`.
    max_log_age : float
        Highest log10(age/yr) column to retain (inclusive).
    T, R, anchor_freq :
        Passed through to `normalised_anchor_value`.

    Returns
    -------
    dict[str, float]
        {age_key: normalised L_nu at anchor_freq}.
    """
    age_cols = [c for c in sed.columns if c != 'WL'
                and float(c) <= max_log_age + 1e-9]
    return {
        age_key: normalised_anchor_value(
            sed, age_key, T=T, R=R, anchor_freq=anchor_freq,
        )
        for age_key in age_cols
    }

""" __________________________ BATCH FILE LOADING __________________________ """
def load_bpass_directory(input_dir, pattern='spectra-bin-imf135_300.z*.dat'):
    """
    Load every BPASS SED file matching `pattern` in `input_dir`.

    Returns
    -------
    seds : dict[str, pandas.DataFrame]
        Filename -> loaded SED.
    Zs : dict[str, float]
        Filename -> metallicity.
    """
    files = sorted(glob(str(Path(input_dir) / pattern)))
    if not files:
        raise FileNotFoundError(
            f'No files matching {pattern} in {input_dir}'
        )

    seds = {}
    Zs = {}
    for f in files:
        name = Path(f).name
        seds[name] = load_sed(f)
        Zs[name] = metallicity_from_filename(name)
    return seds, Zs


def collect_anchor_values(seds, max_log_age=7.0,
                          T=1e5 * u.K, R=cst.R_sun,
                          anchor_freq=FREQ_HI):
    """
    Compute anchor values for every (file, age) pair.

    Returns
    -------
    dict[str, dict[str, float]]
        filename -> {age_key: normalised L_nu at anchor}.
    """
    return {
        name: anchor_values_for_file(
            sed, max_log_age=max_log_age,
            T=T, R=R, anchor_freq=anchor_freq,
        )
        for name, sed in seds.items()
    }

def find_reference_file(Zs, reference_Z=1e-5):
    """
    Pick the filename whose metallicity is closest to `reference_Z`.

    Parameters
    ----------
    Zs : dict[str, float]
        Filename -> metallicity.
    reference_Z : float

    Returns
    -------
    str
        Filename of the reference (lowest / chosen) metallicity.
    """
    return min(Zs, key=lambda n: abs(Zs[n] - reference_Z))

""" __________________________ ALPHA COMPUTATION __________________________ """
def compute_alpha_for_file(file_anchors, ref_anchors):
    """
    Compute alpha(Z, t) = bpass_norm(Z, t, nu_a) / bpass_norm(Z0, t, nu_a)
    for every age key in `file_anchors`.

    Parameters
    ----------
    file_anchors : dict[str, float]
        Anchor values for the current file: {age_key: value}.
    ref_anchors : dict[str, float]
        Anchor values for the reference (Z0) file.

    Returns
    -------
    dict[str, float]
        {age_key: alpha}.
    """
    out = {}
    for age_key, num in file_anchors.items():
        den = ref_anchors.get(age_key)
        if den is None or den == 0:
            raise ValueError(
                f'Reference metallicity has no usable value at age '
                f'{age_key} (den={den}); cannot form ratio.'
            )
        out[age_key] = num / den
    return out


def compute_all_alphas(anchor_vals, ref_name):
    """
    Apply `compute_alpha_for_file` to every file.

    Parameters
    ----------
    anchor_vals : dict[str, dict[str, float]]
        filename -> {age_key: anchor value}.
    ref_name : str
        Filename of the reference metallicity.

    Returns
    -------
    dict[str, dict[str, float]]
        filename -> {age_key: alpha}.
    """
    ref_anchors = anchor_vals[ref_name]
    return {
        name: compute_alpha_for_file(file_anchors, ref_anchors)
        for name, file_anchors in anchor_vals.items()
    }

""" __________________________ BB SPECTRUM CONSTRUCTION __________________________ """
def rescaled_bb_lum_lamb(freq, alpha, T=1e5 * u.K, R=cst.R_sun):
    """
    Build the rescaled black body spectrum on a given frequency grid.

        rescaled_BB(nu) = alpha * B_nu(T, R)

    Returned in L_lambda units on the corresponding wavelength grid.

    Parameters
    ----------
    freq : Quantity [Hz]
    alpha : float
        Multiplicative scale factor.
    T, R : Quantity
        Black body parameters.

    Returns
    -------
    lum_lamb : Quantity [Lsun / AA]
    """
    raw_bb = monochromatic_luminosity(freq, T=T, R=R)
    rescaled = (alpha * raw_bb).to(u.Lsun / u.Hz)
    lum_lamb, _ = lum_freq_to_lum_lamb(rescaled, freq)
    return lum_lamb.to(u.Lsun / u.AA)

def build_rescaled_dataframe(sed, alphas, T=1e5 * u.K, R=cst.R_sun):
    """
    Build the output DataFrame for one file: same wavelength column as
    the input SED, plus one rescaled-BB L_lambda column per age key in
    `alphas`.

    Parameters
    ----------
    sed : pandas.DataFrame
        Input SED (provides the wavelength grid; age columns are
        replaced).
    alphas : dict[str, float]
        {age_key: alpha} for this file.
    T, R : Quantity
        Black body parameters.

    Returns
    -------
    pandas.DataFrame
        Columns: 'WL' followed by each age_key in `alphas`.
    """
    out = pd.DataFrame({'WL': sed['WL'].to_numpy()})
    for age_key, alpha in alphas.items():
        freq, _ = sed_to_lum_freq(sed, age_key)   # frequency grid only
        lum_lamb = rescaled_bb_lum_lamb(freq, alpha, T=T, R=R)
        out[age_key] = lum_lamb.value
    return out

""" __________________________ TOP LEVEL FILE WRITING __________________________ """

def write_sed_file(df, path, fmt='%.6e'):
    """
    Write a DataFrame in BPASS SED format: whitespace-separated, no
    header, no index. Wavelength column is written first.
    """
    cols = ['WL'] + [c for c in df.columns if c != 'WL']
    df[cols].to_csv(path, sep=' ', header=False, index=False,
                    float_format=fmt)
    
def save_rescaled_bb_directory(
    input_dir,
    output_dir,
    pattern='spectra-bin-imf135_300.z*.dat',
    reference_Z=1e-5,
    T=1e5 * u.K,
    R=cst.R_sun,
    anchor_freq=FREQ_HI,
    max_log_age=7.0,
    fmt='%.6e',
    verbose=True,
):
    """
    Build rescaled-BB SED files for every BPASS file in `input_dir`
    matching `pattern`, and write them to `output_dir` in BPASS format
    (whitespace-separated, no header, no index, wavelength column
    first).

    The rescaling enforces, for each age column,

        alpha(Z, t) = bpass_norm(Z,  t, nu_anchor)
                      / bpass_norm(Z0, t, nu_anchor)

    where Z0 is the metallicity closest to `reference_Z`. At Z = Z0
    the rescaled BB equals the raw (T, R) black body.

    Parameters
    ----------
    input_dir, output_dir : str or Path
        Source and destination directories. `output_dir` is created if
        it does not exist.
    pattern : str
        Glob pattern for selecting BPASS files inside `input_dir`.
    reference_Z : float
        Reference metallicity used to anchor alpha.
    T, R : Quantity
        Black body parameters.
    anchor_freq : Quantity [Hz]
        Frequency at which the cross-metallicity ratio is enforced.
    max_log_age : float
        Highest log10(age/yr) column to retain (inclusive).
    fmt : str
        printf-style float format passed through to pandas.to_csv.
    verbose : bool
        If True, print progress.

    Returns
    -------
    alpha_table : pandas.DataFrame
        Rows = filenames, columns = age keys + 'Z',
        values = alpha(Z, t).
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load every BPASS file once
    seds, Zs = load_bpass_directory(in_path, pattern=pattern)
    if verbose:
        print(f'Found {len(seds)} BPASS file(s) in {in_path}')

    # 2. Compute normalised anchor values for every (file, age)
    anchor_vals = collect_anchor_values(
        seds, max_log_age=max_log_age,
        T=T, R=R, anchor_freq=anchor_freq,
    )

    # 3. Pick the reference metallicity and compute alpha(Z, t)
    ref_name = find_reference_file(Zs, reference_Z=reference_Z)
    if verbose:
        print(f'Reference: Z = {Zs[ref_name]:.5g}  ({ref_name})')
    alphas_per_file = compute_all_alphas(anchor_vals, ref_name)

    # 4. Build rescaled-BB DataFrames and write each one to disk
    for name, sed in seds.items():
        out_df = build_rescaled_dataframe(
            sed, alphas_per_file[name], T=T, R=R,
        )
        write_sed_file(out_df, out_path / name, fmt=fmt)
        if verbose:
            print(f'  wrote {name}  '
                  f'(Z = {Zs[name]:.5g}, '
                  f'{len(alphas_per_file[name])} age columns)')

    # 5. Tidy alphas into an inspectable DataFrame
    alpha_table = pd.DataFrame(alphas_per_file).T
    alpha_table['Z'] = [Zs[n] for n in alpha_table.index]
    alpha_table.index.name = 'filename'
    return alpha_table
