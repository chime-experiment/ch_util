"""Channel quality monitor routines"""

import numpy as np
import copy

import caput.time as ctime

from ch_ephem.observers import chime
import ch_ephem.sources

from chimedb import data_index
from ch_util import finder

# Corrections to transit times due to 2deg rotation of cylinders:
TTCORR = {"CygA": -94.4, "CasA": 152.3, "TauA": -236.9, "VirA": -294.5}

CR = 2.0 * np.pi / 180.0  # Cylinder rotation in radians
R = np.array(
    [[np.cos(CR), -np.sin(CR)], [np.sin(CR), np.cos(CR)]]
)  # Cylinder rotation matrix
C = 2.9979e8
PHI = chime.latitude * np.pi / 180.0  # DRAO Latitue
SD = 24.0 * 3600.0 * ctime.SIDEREAL_S  # Sidereal day

_DEFAULT_NODE_SPOOF = {"scinet_online": "/scratch/k/krs/jrs65/chime/archive/online/"}
# _DEFAULT_NODE_SPOOF = {'gong': '/mnt/gong/archive'} # For tests on Marimba


class FeedLocator(object):
    """This class contains functions that do all the computations to
    determine feed positions from data. It also determines the quality
    of data and returns a list of good inputs and frequencies.

    Uppon initialization, it receives visibility data around one or two
    bright sources transits as well as corresponding meta-data.

    Parameters
    ----------
    vis1, [vis2] : Visibility data around bright source transit
    tm1, [tm2] : Timestamp corresponding to vis1 [vis2]
    src1, [src2] : Ephemeris astronomical object corresponding to the
                   transit in vis1 [vis2]
    freqs : frequency axis of vis1 [and vis2]
    prods : Product axis of vis1 [and vis2]
    inputs : inputs loaded in vis1 [and vis2]
    pstns0 : positions of inputs as obtained from the layout database
    bsipts : base inputs used to determine cross correlations loaded
             (might become unecessary in the future)
    """

    def __init__(
        self,
        vis1,
        vis2,
        tm1,
        tm2,
        src1,
        src2,
        freqs,
        prods,
        inputs,
        pstns0,
        bsipts=None,
    ):
        """Basic initialization method"""
        self.adj_freqs = False  # frequencies adjacent in data? (used for some tests)
        self.VERBOSE = False
        self.PATH = True  # True if Pathfinder, False if CHIME

        self.vis1 = vis1
        self.vis2 = vis2
        self.tm1 = tm1
        self.tm2 = tm2
        self.freqs = freqs
        self.prods = prods
        self.inputs = inputs
        self.inprds = np.array(
            [[self.inputs[prod[0]], self.inputs[prod[1]]] for prod in self.prods]
        )
        self.pstns0 = pstns0
        self.bsipts = bsipts

        self.Nfr = len(freqs)
        self.Npr = len(prods)
        self.Nip = len(inputs)
        self.c_bslns0, self.bslns0 = self.set_bslns0()

        self.source1 = src1
        self.source2 = src2
        self.dec1 = self.source1._dec
        if self.source2 is not None:
            self.dec2 = self.source2._dec
        self.tt1 = (
            chime.transit_times(self.source1, self.tm1[0], self.tm1[-1])[0]
            + TTCORR[self.source1.name]
        )
        if self.source2 is not None:
            self.tt2 = (
                chime.transit_times(self.source2, self.tm2[0], self.tm2[-1])[0]
                + TTCORR[self.source2.name]
            )

        # Results place holders:
        self.pass_xd1, self.pass_xd2 = None, None
        self.pass_cont1, self.pass_cont2 = None, None
        self.good_prods = None
        self.good_freqs = None
        self.good_prods_cons = None
        self.good_freqs_cons = None

        # TODO: delete all incidences of self.dists_computed. Subs by
        # initializing here to self.xdists1 = None, etc..
        self.dists_computed = False
        # TODO: this might not be used...
        # Possible to scramble expected baselines. For debug purposes only:
        self.bslins0_scrambled = False

    def set_bslns0(self):
        """ """
        prods = self.prods
        pstns0 = self.pstns0

        c_bslns0 = np.zeros((self.Npr, 2))
        bslns0 = np.zeros((self.Npr, 2))
        for ii, prd in enumerate(prods):
            x0 = pstns0[prd[0], 0]
            y0 = pstns0[prd[0], 1]
            x1 = pstns0[prd[1], 0]
            y1 = pstns0[prd[1], 1]

            # Expected baselines (cylinder coords):
            c_bslns0[ii] = np.array([x0 - x1, y0 - y1])
            # Expected rotated baselines (Earth coords):
            bslns0[ii] = np.dot(R, c_bslns0[ii])

        return c_bslns0, bslns0

    def phase_trans(self, vis, tm, tt):
        """ """
        ph = np.zeros((self.Nfr, self.Npr))
        tridx = np.argmin(abs(tm - tt))  # Transit index

        # TODO: Could add a test to check if transit time falls
        # in between time points (rather than close to a particular point)
        # and take a wheighted average of the two points in this case:
        for ii in range(self.Nfr):
            for jj in range(self.Npr):
                ph[ii, jj] = np.angle(vis[ii, jj, tridx])

        return ph

    def getphases_tr(self):
        """ """
        self.ph1 = self.phase_trans(self.vis1, self.tm1, self.tt1)
        self.ph2 = self.phase_trans(self.vis2, self.tm2, self.tt2)

        return self.ph1, self.ph2

    def get_c_ydist(
        self, ph1=None, ph2=None, good_freqs=None, tol=1.5, Nmax=20
    ):  # knl=3,dfacts=[5,11,17],rms_tol=1.):#,stptol=1.5,stpdev=1.):
        """N-S. Absolutelly assumes contiguous frequencies!!!"""
        # TODO: add test for contiguous freqs!!!

        def yparams(xx, yy):
            """ """
            a, b = np.polyfit(xx, yy, 1)

            ydiff = np.diff(yy, axis=0)
            xdiff = np.diff(xx)
            absydiff = abs(ydiff)

            m, mn = np.zeros(yy.shape[1]), np.zeros(yy.shape[1])
            for ii in range(ydiff.shape[1]):
                slct = absydiff[:, ii] < 2.0 * np.pi - tol
                m[ii] = np.nanmedian(ydiff[slct, ii] / xdiff[slct])
                mn[ii] = np.nanmean(ydiff[slct, ii] / xdiff[slct])

            lines = np.array(
                [
                    (xx - xx[0])[:, np.newaxis] * slp[np.newaxis, :]
                    + yy[0][np.newaxis, :]
                    for slp in [a, m, mn]
                ]
            )
            # Options are:
            # fit-slope-no-unwrap, median-slope-no-unwrap, mean-slope-no-unwrap
            # the same three again with unwrapping and discont=2.*np.pi-tol
            # the same three again with unwrapping and no discont
            y_opts = np.zeros((9, yy.shape[0], yy.shape[1]))
            y_opts[:3] = (yy[np.newaxis, ...] - lines + np.pi) % (2.0 * np.pi) - np.pi
            y_opts[3:6] = np.unwrap(y_opts[:3], discont=2.0 * np.pi - tol, axis=1)
            y_opts[6:] = np.unwrap(y_opts[:3], axis=1)

            # Get goodness of fit:
            y_rms = np.zeros((y_opts.shape[0], y_opts.shape[2]))
            for ii in range(y_opts.shape[0]):
                opt = y_opts[ii]
                a, b = np.polyfit(xx, opt, 1)
                mod = xx[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
                y_rms[ii] = np.nanstd(opt - mod, axis=0)

            y_idx = np.argmin(y_rms, axis=0)

            y_opts[:3] = y_opts[:3] + lines
            y_opts[3:6] = y_opts[3:6] + lines
            y_opts[6:] = y_opts[6:] + lines
            y_nxt = y_opts[
                y_idx[np.newaxis, :],
                np.arange(y_opts.shape[1])[:, np.newaxis],
                np.arange(y_opts.shape[2])[np.newaxis, :],
            ]

            a, b = np.polyfit(xx, y_nxt, 1)

            return y_nxt, a

        if ph1 is None:
            ph1 = self.ph1
        if ph2 is None:
            ph2 = self.ph2

        yslope = (np.copy(ph1 - ph2) + np.pi) % (2.0 * np.pi) - np.pi

        if good_freqs is not None:
            yslope = yslope[good_freqs]
            fr = np.copy(self.freqs[good_freqs])

        yslope = np.unwrap(yslope, discont=2.0 * np.pi - tol, axis=0)
        yslope = yslope - yslope[0]

        yslp, a_prev = yparams(fr, yslope)  # First iteration
        for ii in range(Nmax):
            yslp, a = yparams(fr, yslp)
            a_incr = abs(a - a_prev) / (abs(a + a_prev) * 0.5)
            pass_y = a_incr < 1e-2
            if a_incr.all():
                break
            else:
                a_prev = a

        # TODO: now it's only one per chan. Change rotation coda appropriatelly
        c_ydists = (
            a
            / 1e6
            * C
            / (
                2.0
                * np.pi
                * (
                    np.cos(np.pi / 2.0 - self.dec1 + PHI)
                    - np.cos(np.pi / 2.0 - self.dec2 + PHI)
                )
            )
        )

        return c_ydists

    def get_c_ydist_perfreq(self, ph1=None, ph2=None):
        """Old N-S dists function. TO be used only in case a continuum of
        frequencies is not available
        """

        if ph1 is None:
            ph1 = self.ph1
        if ph2 is None:
            ph2 = self.ph2

        c_ydists0 = self.c_bslns0[:, 1]

        dist_period = abs(
            C
            / (
                self.freqs
                * 1e6
                * (
                    np.cos(np.pi / 2.0 - self.dec1 + PHI)
                    - np.cos(np.pi / 2.0 - self.dec2 + PHI)
                )
            )
        )
        exp_dist = c_ydists0[np.newaxis, :] % dist_period[:, np.newaxis]
        base_ctr = c_ydists0[np.newaxis, :] - exp_dist
        base_up = base_ctr + dist_period[:, np.newaxis]
        base_down = base_ctr - dist_period[:, np.newaxis]

        phdiff = ph1 - ph2
        c_ydists = (
            phdiff
            * C
            / (
                2.0
                * np.pi
                * self.freqs[:, np.newaxis]
                * 1e6
                * (
                    np.cos(np.pi / 2.0 - self.dec1 + PHI)
                    - np.cos(np.pi / 2.0 - self.dec2 + PHI)
                )
            )
        )
        c_ydists = c_ydists % dist_period[:, np.newaxis]
        dist_opts = np.array(
            [c_ydists + base_up, c_ydists + base_ctr, c_ydists + base_down]
        )
        idxs = np.argmin(abs(dist_opts - c_ydists0[np.newaxis, np.newaxis, :]), axis=0)
        c_ydists = np.array(
            [
                [dist_opts[idxs[ii, jj], ii, jj] for jj in range(self.Npr)]
                for ii in range(self.Nfr)
            ]
        )

        return c_ydists

    # TODO: change to 'yparams'
    def params_ft(self, tm, vis, dec, x0_shift=5.0):
        """Extract relevant parameters from source transit
            visibility in two steps:
                1) FFT visibility
                2) Fit a gaussian to the transform

        Parameters
        ----------
        tm : array-like
            Independent variable (time)
        trace : array-like
            Dependent variable (visibility)
        freq : float
            Frenquency of the visibility trace, in MHz.
        dec : float
            Declination of source. Used for initial guess of
            gaussian width. Defaults to CygA declination: 0.71

        Returns
        -------
        popt : array of float
            List with optimal parameters: [A,mu,sig2]
        pcov : array of float
            Covariance matrix for optimal parameters.
            For details see documentation on numpy.curve_fit

        """
        from scipy.optimize import curve_fit

        freqs = self.freqs
        prods = self.prods

        # Gaussian function for fit:
        def gaus(x, A, mu, sig2):
            return A * np.exp(-((x - mu) ** 2) / (2.0 * sig2))

        # FFT:
        # TODO: add check to see if length is multiple of 2
        Nt = len(tm)
        dt = tm[1] - tm[0]
        ft = np.fft.fft(vis, axis=2)
        fr = np.fft.fftfreq(Nt, dt)
        # Re-order frequencies:
        ft_ord = np.concatenate(
            (ft[..., Nt // 2 + Nt % 2 :], ft[..., : Nt // 2 + Nt % 2]), axis=2
        )
        ft_ord = abs(ft_ord)
        fr_ord = np.concatenate((fr[Nt // 2 + Nt % 2 :], fr[: Nt // 2 + Nt % 2]))

        # Gaussian fits:
        # Initial guesses:
        # distx_0 = self.bslns0[:,0] # Should all be either 0 or +-22 for Pathfinder
        x0_shift = 5.0
        distx_0 = self.bslns0[:, 0] + x0_shift  # Shift to test robustness
        mu0 = (
            -2.0
            * np.pi
            * freqs[:, np.newaxis]
            * 1e6
            * distx_0[np.newaxis, :]
            * np.sin(np.pi / 2.0 - dec)
            / (3e8 * SD)
        )
        ctr_idx = np.argmin(
            abs(fr_ord[np.newaxis, np.newaxis, :] - mu0[..., np.newaxis]), axis=2
        )
        A0 = np.array(
            [
                [ft_ord[ii, jj, ctr_idx[ii, jj]] for jj in range(self.Npr)]
                for ii in range(self.Nfr)
            ]
        )
        # 1 deg => dt = 1*(pi/180)*(24*3600/2*pi) = 240s
        sigsqr0 = 1.0 / (4.0 * np.pi**2 * (240.0 * np.cos(dec)) ** 2)
        p0 = np.array(
            [
                [[A0[ii, jj], mu0[ii, jj], sigsqr0] for jj in range(self.Npr)]
                for ii in range(self.Nfr)
            ]
        )
        # Perform fit:
        # TODO: there must be a way to do the fits without the for-loops
        prms = np.zeros((self.Nfr, self.Npr, 3))
        for ii in range(self.Nfr):
            for jj in range(self.Npr):
                try:
                    popt, pcov = curve_fit(gaus, fr_ord, ft_ord[ii, jj, :], p0[ii, jj])
                    prms[ii, jj] = np.array(popt)
                # TODO: look for the right exception:
                except:
                    # TODO: Use masked arrays instead of None?
                    prms[ii, jj] = [None] * 3

        return prms

    # TODO: change to 'get_yparams'
    def getparams_ft(self):
        """ """
        # TODO: Add test to eliminate bad fits!
        self.ft_prms1 = self.params_ft(self.tm1, self.vis1, self.dec1)
        if self.source2 is not None:
            self.ft_prms2 = self.params_ft(self.tm2, self.vis2, self.dec2)
        else:
            self.ft_prms2 = None

        return self.ft_prms1, self.ft_prms2

    # TODO: change all occurences of 'get_xdist' to 'xdists'
    # to make it more consistent
    def get_xdist(self, ft_prms, dec):
        """E-W"""
        xdists = (
            -ft_prms[..., 1]
            * SD
            * C
            / (
                2.0
                * np.pi
                * self.freqs[:, np.newaxis]
                * 1e6
                * np.sin(np.pi / 2.0 - dec)
            )
        )

        return xdists

    def data_quality(self):
        """ """
        if self.pass_xd1 is None:
            if self.source2 is not None:
                self.xdist_test(
                    self.c_xdists1, self.c_xdists2
                )  # Assigns self.pass_xd1, self.pass_xd2
            else:
                # Slightly higher tolerance since it uses rotated dists
                self.xdist_test(self.xdists1, tol=2.5)

        if self.pass_cont1 is None:
            self.continuity_test()  # Assigns self.pass_cont1 and self.pass_cont2

        gpxd1, gfxd1 = self.good_prod_freq(self.pass_xd1)
        gpc1, gfc1 = self.good_prod_freq(self.pass_cont1)

        if self.source2 is not None:
            gpxd2, gfxd2 = self.good_prod_freq(self.pass_xd2)
            gpc2, gfc2 = self.good_prod_freq(self.pass_cont2)

            self.good_prods = np.logical_and(
                np.logical_or(gpc1, gpc2), np.logical_or(gpxd1, gpxd2)
            )  # good prods
            self.good_freqs = np.logical_and(
                np.logical_or(gfc1, gfc2), np.logical_or(gfxd1, gfxd2)
            )  # good freqs

            # TODO: Delete these conservative estimates?
            self.good_prods_cons = np.logical_and(
                np.logical_and(gpc1, gpc2), np.logical_and(gpxd1, gpxd2)
            )  # Conservative good prods
            self.good_freqs_cons = np.logical_and(
                np.logical_and(gfc1, gfc2), np.logical_and(gfxd1, gfxd2)
            )  # Conservative good freqs
        else:
            self.good_prods = np.logical_and(gpc1, gpxd1)  # good prods
            self.good_freqs = np.logical_and(gfc1, gfxd1)  # good freqs

        if self.bsipts is not None:
            self.set_good_ipts(self.bsipts)  # good_prods to good_ipts

    def single_source_test(self):
        """ """
        self.getparams_ft()
        self.xdists1 = self.get_xdist(self.ft_prms1, self.dec1)

        self.data_quality()

    def get_dists(self):
        """ """
        # Get x distances in Earth coords (EW)
        self.getparams_ft()
        self.xdists1 = self.get_xdist(self.ft_prms1, self.dec1)
        self.xdists2 = self.get_xdist(self.ft_prms2, self.dec2)
        # Preliminary test for bad freqs (needed for ydists):
        if self.pass_cont1 is None:
            self.continuity_test()  # Assigns self.pass_cont1 and self.pass_cont2
        gpc1, gfc1 = self.good_prod_freq(self.pass_cont1)
        gpc2, gfc2 = self.good_prod_freq(self.pass_cont2)
        gf = np.logical_and(gfc1, gfc2)  # Preliminary good freqs
        # Get y distances in cylinder coordinates (NS rotated by 2 deg)
        self.getphases_tr()
        self.c_ydists = self.get_c_ydist(good_freqs=gf)
        # Transform between Earth and cylinder coords
        self.c_xdists1 = (
            self.xdists1 + self.c_ydists[np.newaxis, :] * np.sin(CR)
        ) / np.cos(CR)
        self.c_xdists2 = (
            self.xdists2 + self.c_ydists[np.newaxis, :] * np.sin(CR)
        ) / np.cos(CR)
        self.ydists1 = (
            self.xdists1 + self.c_ydists[np.newaxis, :] * np.sin(CR)
        ) * np.tan(CR) + self.c_ydists[np.newaxis, :] * np.cos(CR)
        self.ydists2 = (
            self.xdists2 + self.c_ydists[np.newaxis, :] * np.sin(CR)
        ) * np.tan(CR) + self.c_ydists[np.newaxis, :] * np.cos(CR)

        self.dists_computed = True

        self.data_quality()

        return self.c_xdists1, self.c_xdists2, self.c_ydists

    def set_good_ipts(self, base_ipts):
        """Good_prods to good_ipts"""
        inp_list = [inpt for inpt in self.inputs]  # Full input list
        self.good_ipts = np.zeros(self.inputs.shape, dtype=bool)
        for ii, inprd in enumerate(self.inprds):
            if inprd[0] not in base_ipts:
                self.good_ipts[inp_list.index(inprd[0])] = self.good_prods[ii]
            if inprd[1] not in base_ipts:
                self.good_ipts[inp_list.index(inprd[1])] = self.good_prods[ii]
            if (inprd[0] in base_ipts) and (inprd[1] in base_ipts):
                self.good_ipts[inp_list.index(inprd[0])] = self.good_prods[ii]
                self.good_ipts[inp_list.index(inprd[1])] = self.good_prods[ii]
        # To make sure base inputs are tagged good:
        for bsip in base_ipts:
            self.good_ipts[inp_list.index(bsip)] = True

    def solv_pos(self, dists, base_ipt):
        """ """
        from scipy.linalg import svd

        # Matrix defining order of subtraction for baseline distances
        M = np.zeros((self.Npr, self.Nip - 1))
        # Remove base_ipt as its position will be set to zero
        sht_inp_list = [inpt for inpt in self.inputs if inpt != base_ipt]
        for ii, inprd in enumerate(self.inprds):
            if inprd[0] != base_ipt:
                M[ii, sht_inp_list.index(inprd[0])] = 1.0
            if inprd[1] != base_ipt:
                M[ii, sht_inp_list.index(inprd[1])] = -1.0
        U, s, Vh = svd(M)
        # TODO: add test for small s values to zero. Check existing code for that.
        # Pseudo-inverse:
        psd_inv = np.dot(np.transpose(Vh) * (1.0 / s)[np.newaxis, :], np.transpose(U))
        # Positions:
        pstns = np.dot(psd_inv, dists)
        # Add position of base_input
        inp_list = [inpt for inpt in self.inputs]  # Full input list
        bs_inpt_idx = inp_list.index(base_ipt)  # Original index of base_ipt
        pstns = np.insert(pstns, bs_inpt_idx, 0.0)

        return pstns

    def get_postns(self):
        """ """
        self.c_xd1 = np.nanmedian(self.c_xdists1[self.good_freqs], axis=0)
        self.c_xd2 = np.nanmedian(self.c_xdists2[self.good_freqs], axis=0)
        # Solve positions:
        self.c_y = self.solv_pos(self.c_ydists, self.bsipts[0])
        self.c_x1 = self.solv_pos(self.c_xd1, self.bsipts[0])
        self.c_x2 = self.solv_pos(self.c_xd2, self.bsipts[0])
        self.expy = self.solv_pos(self.c_bslns0[:, 1], self.bsipts[0])
        self.expx = self.solv_pos(self.c_bslns0[:, 0], self.bsipts[0])

        return self.c_x1, self.c_x2, self.c_y

    def xdist_test(self, xds1, xds2=None, tol=2.0):
        """ """

        def get_centre(xdists, tol):
            """Returns the median (across frequencies) of NS separation dists for each
            baseline if this median is withing *tol* of a multiple of 22 meters. Else,
            returns the multiple of 22 meters closest to this median (up to 3*22=66 meters)
            """
            xmeds = np.nanmedian(xdists, axis=0)
            cylseps = np.arange(-1, 2) * 22.0 if self.PATH else np.arange(-3, 4) * 22.0
            devs = abs(xmeds[:, np.newaxis] - cylseps[np.newaxis, :])
            devmins = devs.min(axis=1)
            centres = np.array(
                [
                    (
                        xmeds[ii]  # return median
                        if devmins[ii] < tol  # if reasonable
                        else cylseps[np.argmin(devs[ii])]
                    )  # or use closest value
                    for ii in range(devmins.size)
                ]
            )

            return centres

        xcentre1 = get_centre(xds1, tol)
        xerr1 = abs(xds1 - xcentre1[np.newaxis, :])
        self.pass_xd1 = xerr1 < tol

        if xds2 is not None:
            xcentre2 = get_centre(xds2, tol)
            xerr2 = abs(xds2 - xcentre2[np.newaxis, :])
            self.pass_xd2 = xerr2 < tol
        else:
            self.pass_xd2 = None

        return self.pass_xd1, self.pass_xd2

    def continuity_test(self, tol=0.2, knl=5):
        """Call only if freqs are adjacent.
        Uses xdists (Earth coords) instead of c_xdists (cylinder coords)
        to allow for calling before ydists are computed. Doesn't make any
        difference for this test. Results are used in computing y_dists.
        """
        from scipy.signal import medfilt

        clean_xdists1 = medfilt(self.xdists1, kernel_size=[knl, 1])
        diffs1 = abs(self.xdists1 - clean_xdists1)
        self.pass_cont1 = diffs1 < tol

        if self.source2 is not None:
            clean_xdists2 = medfilt(self.xdists2, kernel_size=[knl, 1])
            diffs2 = abs(self.xdists2 - clean_xdists2)
            self.pass_cont2 = diffs2 < tol
        else:
            self.pass_cont2 = None

        return self.pass_cont1, self.pass_cont2

    def good_prod_freq(
        self, pass_rst, tol_ch1=0.3, tol_ch2=0.7, tol_fr1=0.6, tol_fr2=0.7
    ):
        """Tries to determine overall bad products and overall bad frequencies
        from a test_pass result.
        """

        # First iteration:
        chans_score = np.sum(pass_rst, axis=0) / float(pass_rst.shape[0])
        freqs_score = np.sum(pass_rst, axis=1) / float(pass_rst.shape[1])
        good_chans = chans_score > tol_ch1
        good_freqs = freqs_score > tol_fr1
        # Second Iteration:
        pass_gch = pass_rst[:, np.where(good_chans)[0]]  # Only good channels
        pass_gfr = pass_rst[np.where(good_freqs)[0], :]  # Only good freqs
        chans_score = np.sum(pass_gfr, axis=0) / float(pass_gfr.shape[0])
        freqs_score = np.sum(pass_gch, axis=1) / float(pass_gch.shape[1])
        good_chans = chans_score > tol_ch2
        good_freqs = freqs_score > tol_fr2

        return good_chans, good_freqs


class ChanMonitor(object):
    """This class provides the user interface to FeedLocator.

    It initializes instances of FeedLocator (normally one per polarization)
    and returns results combined lists of results (good channels and positions,
    agreement/disagreement with the layout database, etc.)

    Feed locator should not
    have to sepparate the visibilities in data to run the test on and data not to run the
    test on. ChanMonitor should make the sepparation and provide FeedLocator with the right
    data cube to test.

    Parameters
    ----------
    t1 [t2] : Initial [final] time for the test period. If t2 not provided it is
              set to 1 sideral day after t1
    freq_sel
    prod_sel
    """

    def __init__(
        self,
        t1,
        t2=None,
        freq_sel=None,
        prod_sel=None,
        bswp1=26,
        bswp2=90,
        bsep1=154,
        bsep2=218,
    ):
        """Here t1 and t2 have to be unix time (floats)"""
        self.t1 = t1
        if t2 is None:
            self.t2 = self.t1 + SD
        else:
            self.t2 = t2

        self.acq_list = None
        self.night_acq_list = None

        self.finder = None
        self.night_finder = None

        self.source1 = None
        self.source2 = None

        #        if prod_sel is not None:
        self.prod_sel = prod_sel
        #        if freq_sel is not None:
        self.freq_sel = freq_sel

        self.dat1 = None
        self.dat2 = None
        self.tm1 = None
        self.tm2 = None

        self.freqs = None
        self.prods = None
        self.input_map = None
        self.inputs = None

        self.corr_inputs = None
        self.pwds = None
        self.pstns = None
        self.p1_idx, self.p2_idx = None, None

        self.bswp1 = bswp1
        self.bsep1 = bsep1
        self.bswp2 = bswp2
        self.bsep2 = bsep2

    @classmethod
    def fromdate(
        cls,
        date,
        freq_sel=None,
        prod_sel=None,
        bswp1=26,
        bswp2=90,
        bsep1=154,
        bsep2=218,
    ):
        """Initialize class from date"""
        t1 = ctime.datetime_to_unix(date)
        return cls(
            t1,
            freq_sel=freq_sel,
            prod_sel=prod_sel,
            bswp1=bswp1,
            bswp2=bswp2,
            bsep1=bsep1,
            bsep2=bsep2,
        )

    # TODO: this is kind of silly right now.
    # If it is initialized from data, I should use the data given
    # or not allow for that possibility.
    @classmethod
    def fromdata(cls, data, freq_sel=None, prod_sel=None):
        """Initialize class from andata object"""
        t1 = data.time[0]
        t2 = data.time[-1]
        return cls(t1, t2, freq_sel=freq_sel, prod_sel=prod_sel)

    # TODO: test for adjacent freqs to pass to FeedLocator

    def get_src_cndts(self):
        """ """
        clr, ntt, srcs = self.get_sunfree_srcs()

        grd_dict = {"CygA": 4, "CasA": 4, "TauA": 3, "VirA": 1}
        # Grades for each source
        grds = [
            (
                grd_dict[src.name] - 2
                if (
                    (src.name in ["CygA", "CasA"]) and (not ntt[ii])
                )  # CasA and CygA at daytime worse than TauA at night
                else grd_dict[src.name]
            )
            for ii, src in enumerate(srcs)
        ]

        # Grade 0 if not clear of Sun:
        grds = [grd if clr[ii] else 0 for ii, grd in enumerate(grds)]

        # Source candidates ordered in decreasing quality
        src_cndts = [
            src
            for grd, src in sorted(
                zip(grds, srcs), key=lambda entry: entry[0], reverse=True
            )
            if grd != 0
        ]

        return src_cndts

    def get_pol_prod_idx(self, pol_inpt_idx):
        """ """
        pol_prod_idx = []
        for ii, prd in enumerate(self.prods):
            if (prd[0] in pol_inpt_idx) and (prd[1] in pol_inpt_idx):
                pol_prod_idx.append(ii)

        return pol_prod_idx

    def get_feedlocator(self, pol=1):
        """ """
        if pol == 1:
            pol_inpt_idx = self.p1_idx
            bsipts = [self.bswp1, self.bsep1]
        elif pol == 2:
            pol_inpt_idx = self.p2_idx
            bsipts = [self.bswp2, self.bsep2]

        pol_prod_idx = self.get_pol_prod_idx(pol_inpt_idx)

        inputs = self.inputs[pol_inpt_idx]
        pstns = self.pstns[pol_inpt_idx]
        prods = []
        for prd in self.prods[pol_prod_idx]:
            idx0 = np.where(inputs == self.inputs[prd[0]])[0][0]
            idx1 = np.where(inputs == self.inputs[prd[1]])[0][0]
            prods.append((idx0, idx1))

        if self.source2 is not None:
            fl = FeedLocator(
                self.dat1.vis[:, pol_prod_idx, :],
                self.dat2.vis[:, pol_prod_idx, :],
                self.tm1,
                self.tm2,
                self.source1,
                self.source2,
                self.freqs,
                prods,
                inputs,
                pstns,
                bsipts,
            )
        else:
            fl = FeedLocator(
                self.dat1.vis[:, pol_prod_idx, :],
                None,
                self.tm1,
                None,
                self.source1,
                self.source2,
                self.freqs,
                prods,
                inputs,
                pstns,
                bsipts,
            )

        return fl

    def init_feedloc_p1(self):
        """ """
        self.flp1 = self.get_feedlocator()
        return self.flp1

    def init_feedloc_p2(self):
        """ """
        self.flp2 = self.get_feedlocator(pol=2)
        return self.flp2

    def get_cyl_pol(self, corr_inputs, pwds):
        """ """
        wchp1, wchp2, echp1, echp2 = [], [], [], []
        for ii, inpt in enumerate(corr_inputs):
            if pwds[ii]:
                if inpt.reflector == "W_cylinder":
                    if inpt.pol == "S":
                        wchp1.append(ii)
                    else:
                        wchp2.append(ii)
                elif inpt.reflector == "E_cylinder":
                    if inpt.pol == "S":
                        echp1.append(ii)
                    else:
                        echp2.append(ii)
                else:
                    # This probably doesn't happen...
                    pass
            else:
                # TODO: this only makes sense for the pathfinder:
                if ii < 64:
                    wchp1.append(ii)
                elif ii < 128:
                    wchp2.append(ii)
                elif ii < 192:
                    echp1.append(ii)
                else:
                    echp2.append(ii)

        return [wchp1, wchp2, echp1, echp2]

    def get_pos_pol(self, corr_inputs, pwds):
        """ """
        Ninpts = len(pwds)
        p1_idx, p2_idx = [], []
        pstns = np.zeros((Ninpts, 2))  # In-cylinder positions
        for ii, inpt in enumerate(corr_inputs):
            if pwds[ii]:
                pstns[ii, 0] = inpt.cyl * 22.0
                pstns[ii, 1] = -1.0 * inpt.pos
                if inpt.pol == "S":
                    p1_idx.append(ii)
                else:
                    p2_idx.append(ii)
            else:
                # TODO: this only makes sense for the pathfinder:
                # Numbers were taken from layout database
                if ii < 64:
                    pstns[ii, 0] = 0.0
                    pstns[ii, 1] = -8.767 - 0.3048 * float(ii)
                    p1_idx.append(ii)
                elif ii < 128:
                    pstns[ii, 0] = 0.0
                    pstns[ii, 1] = -8.767 - 0.3048 * float(ii - 64)
                    p2_idx.append(ii)
                elif ii < 192:
                    pstns[ii, 0] = 22.0
                    pstns[ii, 1] = -8.7124 - 0.3048 * float(ii - 128)
                    p1_idx.append(ii)
                else:
                    pstns[ii, 0] = 22.0
                    pstns[ii, 1] = -8.7124 - 0.3048 * float(ii - 192)
                    p2_idx.append(ii)

        return pstns, p1_idx, p2_idx

    def set_metadata(self, tms, input_map):
        """Sets self.corr_inputs, self.pwds, self.pstns, self.p1_idx, self.p2_idx"""
        from ch_util import tools

        # Get CHIME ON channels:
        half_time = ctime.unix_to_datetime(tms[int(len(tms) // 2)])
        corr_inputs = tools.get_correlator_inputs(half_time)
        self.corr_inputs = tools.reorder_correlator_inputs(input_map, corr_inputs)
        pwds = tools.is_chime_on(self.corr_inputs)  # Which inputs are CHIME ON antennas
        self.pwds = np.array(pwds, dtype=bool)
        # Get cylinders and polarizations
        self.pstns, self.p1_idx, self.p2_idx = self.get_pos_pol(
            self.corr_inputs, self.pwds
        )

    def determine_bad_gpu_nodes(self, data, frac_time_on=0.7):
        node_on = np.any(data.vis[:].real != 0.0, axis=1)

        self.gpu_node_flag = np.sum(node_on, axis=1) > frac_time_on * node_on.shape[1]

    def get_prod_sel(self, data):
        """ """
        from ch_util import tools

        input_map = data.input
        tms = data.time
        half_time = ctime.unix_to_datetime(tms[int(len(tms) // 2)])
        corr_inputs = tools.get_correlator_inputs(half_time)
        corr_inputs = tools.reorder_correlator_inputs(input_map, corr_inputs)
        pwds = tools.is_chime_on(corr_inputs)  # Which inputs are CHIME ON antennas

        wchp1, wchp2, echp1, echp2 = self.get_cyl_pol(corr_inputs, pwds)

        # Ensure base channels are CHIME and ON
        while not pwds[np.where(input_map["chan_id"] == self.bswp1)[0][0]]:
            self.bswp1 += 1
        while not pwds[np.where(input_map["chan_id"] == self.bswp2)[0][0]]:
            self.bswp2 += 1
        while not pwds[np.where(input_map["chan_id"] == self.bsep1)[0][0]]:
            self.bsep1 += 1
        while not pwds[np.where(input_map["chan_id"] == self.bsep2)[0][0]]:
            self.bsep2 += 1

        prod_sel = []
        for ii, prod in enumerate(data.prod):
            add_prod = False
            add_prod = add_prod or (
                (prod[0] == self.bswp1 and prod[1] in echp1)
                or (prod[1] == self.bswp1 and prod[0] in echp1)
            )
            add_prod = add_prod or (
                (prod[0] == self.bswp2 and prod[1] in echp2)
                or (prod[1] == self.bswp2 and prod[0] in echp2)
            )
            add_prod = add_prod or (
                (prod[0] == self.bsep1 and prod[1] in wchp1)
                or (prod[1] == self.bsep1 and prod[0] in wchp1)
            )
            add_prod = add_prod or (
                (prod[0] == self.bsep2 and prod[1] in wchp2)
                or (prod[1] == self.bsep2 and prod[0] in wchp2)
            )

            if add_prod:
                prod_sel.append(ii)

        prod_sel.sort()

        return prod_sel, pwds

    def get_data(self):
        """ """
        from ch_util import ni_utils

        self.set_acq_list()
        src_cndts = self.get_src_cndts()

        for src in src_cndts:
            results_list = self.get_results(src)
            if len(results_list) != 0:
                if self.source1 is None:
                    # Get prod_sel if not given:
                    if self.prod_sel is None:
                        # Load data with a single frequency to get prod_sel
                        dat = results_list[0].as_loaded_data(freq_sel=[0])
                        self.prod_sel, pwds = self.get_prod_sel(dat)
                    # Load data:
                    self.source1 = src
                    self.dat1 = results_list[0].as_loaded_data(
                        prod_sel=self.prod_sel, freq_sel=self.freq_sel
                    )
                    # TODO: correct process_synced_data to not crash when no NS
                    try:
                        self.dat1 = ni_utils.process_synced_data(self.dat1)
                    except:
                        pass
                    self.freqs = self.dat1.freq
                    self.prods = self.dat1.prod
                    self.input_map = self.dat1.input
                    self.inputs = self.input_map["chan_id"]
                    self.tm1 = self.dat1.time
                    # Set metadata (corr_inputs, pstns, polarizatins, etc...
                    self.set_metadata(self.tm1, self.input_map)

                    # Determine what frequencies are bad
                    # due to gpu nodes that are down
                    self.determine_bad_gpu_nodes(self.dat1)

                # TODO: get corr_inputs for dat2 as well and compare to dat1
                elif self.source2 is None:
                    self.source2 = src
                    self.dat2 = results_list[0].as_loaded_data(
                        prod_sel=self.prod_sel, freq_sel=self.freq_sel
                    )
                    # TODO: correct process_synced_data to not crash when no NS
                    try:
                        self.dat2 = ni_utils.process_synced_data(self.dat2)
                    except:
                        pass
                    self.tm2 = self.dat2.time
                    break

        return self.source1, self.source2

    def get_results(self, src, tdelt=2800):
        """If self.finder exists, then it takes a deep copy of this object,
        further restricts the time range to include only src transits,
        and then queries the database to obtain a list of the acquisitions.
        If self.finder does not exist, then it creates a finder object,
        restricts the time range to include only src transits between
        self.t1 and self.t2, and then queries the database to obtain a list
        of the acquisitions.
        """

        if self.finder is not None:
            f = copy.deepcopy(self.finder)
        else:
            f = finder.Finder(node_spoof=_DEFAULT_NODE_SPOOF)
            f.filter_acqs((data_index.ArchiveInst.name == "pathfinder"))
            f.only_corr()
            f.set_time_range(self.t1, self.t2)

        f.include_transits(src, time_delta=tdelt)

        return f.get_results()

    def set_acq_list(self):
        """This method sets four attributes.  The first two attributes
        are 'night_finder' and 'night_acq_list', which are the
        finder object and list of acquisitions that
        contain all night time data between self.t1 and self.t2.
        The second two attributes are 'finder' and 'acq_list',
        which are the finder object and list of acquisitions
        that contain all data beween self.t1 and self.t2 with the
        sunrise, sun transit, and sunset removed.
        """

        # Create a Finder object and focus on time range
        f = finder.Finder(node_spoof=_DEFAULT_NODE_SPOOF)
        f.filter_acqs((data_index.ArchiveInst.name == "pathfinder"))
        f.only_corr()
        f.set_time_range(self.t1, self.t2)

        # Create a list of acquisitions that only contain data collected at night
        f_night = copy.deepcopy(f)
        f_night.exclude_daytime()

        self.night_finder = f_night
        self.night_acq_list = f_night.get_results()

        # Create a list of acquisitions that flag out sunrise, sun transit, and sunset
        mm = ctime.unix_to_datetime(self.t1).month
        dd = ctime.unix_to_datetime(self.t1).day
        mm = mm + float(dd) / 30.0

        fct = 3.0
        tol1 = (np.arctan((mm - 3.0) * fct) + np.pi / 2.0) * 10500.0 / np.pi + 1500.0
        tol2 = (np.pi / 2.0 - np.arctan((mm - 11.0) * fct)) * 10500.0 / np.pi + 1500.0
        ttol = np.minimum(tol1, tol2)

        fct = 5.0
        tol1 = (np.arctan((mm - 4.0) * fct) + np.pi / 2.0) * 2100.0 / np.pi + 6000.0
        tol2 = (np.pi / 2.0 - np.arctan((mm - 10.0) * fct)) * 2100.0 / np.pi + 6000.0
        rstol = np.minimum(tol1, tol2)

        f.exclude_sun(time_delta=ttol, time_delta_rise_set=rstol)

        self.finder = f
        self.acq_list = f.get_results()

    def get_sunfree_srcs(self, srcs=None):
        """This method uses the attributes 'night_acq_list' and
        'acq_list' to determine the srcs that transit
        in the available data.  If these attributes do not
        exist, then the method 'set_acq_list' is called.
        If srcs is not specified, then it defaults to the
        brightest four radio point sources in the sky:
        CygA, CasA, TauA, and VirA.
        """

        if self.acq_list is None:
            self.set_acq_list()

        if srcs is None:
            srcs = [
                ch_ephem.sources.CygA,
                ch_ephem.sources.CasA,
                ch_ephem.sources.TauA,
                ch_ephem.sources.VirA,
            ]
        Ns = len(srcs)

        clr = [False] * Ns
        ntt = [False] * Ns  # night transit

        for ii, src in enumerate(srcs):
            night_transit = np.array([])
            for acq in self.night_acq_list:
                night_transit = np.append(
                    night_transit, chime.transit_times(src, *acq[1])
                )

            if night_transit.size:
                ntt[ii] = True

            if src.name in ["CygA", "CasA"]:
                transit = np.array([])
                for acq in self.acq_list:
                    transit = np.append(transit, chime.transit_times(src, *acq[1]))

                if transit.size:
                    clr[ii] = True

            else:
                clr[ii] = ntt[ii]

        return clr, ntt, srcs

    def single_source_check(self):
        """Assumes self.source1 is NOT None"""
        Nipts = len(self.inputs)
        self.good_ipts = np.zeros(Nipts, dtype=bool)
        self.good_freqs = None

        if len(self.p1_idx) > 0:
            self.init_feedloc_p1()  # Initiate FeedLocator
            self.flp1.single_source_test()
            self.good_freqs = self.flp1.good_freqs
            good_frac = self.get_res_sing_src(self.flp1)
            if good_frac < 0.6:
                msg = """
WARNING!
Less than 60% of P1 channels turned out good.
This may be due to a poor choice of base channel.
Consider re-running the test with different
bswp1 and bsep1 arguments
"""
                print(msg)

        if len(self.p2_idx) > 0:
            self.init_feedloc_p2()  # Initiate FeedLocator
            self.flp2.single_source_test()
            good_frac = self.get_res_sing_src(self.flp2)
            if good_frac < 0.6:
                msg = """
WARNING!
Less than 60% of P2 channels turned out good.
This may be due to a poor choice of base channel.
Consider re-running the test with different
bswp2 and bsep2 arguments
"""
                print(msg)

            if self.good_freqs is None:
                self.good_freqs = self.flp2.good_freqs
            else:
                self.good_freqs = np.logical_or(self.good_freqs, self.flp2.good_freqs)

        self.results_summary()

    def full_check(self):
        """ """
        if self.source1 is None:
            self.get_data()
        if self.source2 is None:
            if self.source1 is None:
                raise RuntimeError("No sources available.")
            else:
                self.single_source_check()
        else:
            Nipts = len(self.inputs)
            self.good_ipts = np.zeros(Nipts, dtype=bool)
            self.postns = np.zeros((Nipts, 2))
            self.expostns = np.zeros((Nipts, 2))
            self.good_freqs = None
            if len(self.p1_idx) > 0:
                self.init_feedloc_p1()  # Initiate FeedLocator
                self.flp1.get_dists()  # Run tests
                self.flp1.get_postns()  # Solve for positions

                self.good_freqs = self.flp1.good_freqs
                good_frac = self.get_test_res(self.flp1)
                if good_frac < 0.6:
                    msg = """
WARNING!
Less than 60% of P1 channels turned out good.
This may be due to a poor choice of base channel.
Consider re-running the test with different
bswp1 and bsep1 arguments
"""
                    print(msg)

            if len(self.p2_idx) > 0:
                self.init_feedloc_p2()  # Initiate FeedLocator
                self.flp2.get_dists()  # Run tests
                self.flp2.get_postns()  # Solve for positions

                if self.good_freqs is None:
                    self.good_freqs = self.flp2.good_freqs
                else:
                    self.good_freqs = np.logical_or(
                        self.good_freqs, self.flp2.good_freqs
                    )
                good_frac = self.get_test_res(self.flp2)
                if good_frac < 0.6:
                    msg = """
WARNING!
Less than 60% of P2 channels turned out good.
This may be due to a poor choice of base channel.
Consider re-running the test with different
bswp2 and bsep2 arguments
"""
                    print(msg)

            self.results_summary()

    def results_summary(self):
        """ """
        self.bad_ipts = self.input_map[np.logical_not(self.good_ipts)]
        self.deemed_bad_but_good = self.input_map[
            np.logical_and(np.logical_not(self.pwds), self.good_ipts)
        ]
        self.bad_not_accounted = self.input_map[
            np.logical_and(self.pwds, np.logical_not(self.good_ipts))
        ]
        if self.source2 is not None:
            # TODO: maybe use only x-position. Y is too erratic...
            self.pos_err = np.sum((self.postns - self.expostns) ** 2, axis=1) ** 0.5
            self.wrong_position = self.input_map[self.pos_err > 1.0]

    def get_test_res(self, fl):
        """ """
        for ii, ipt in enumerate(self.inputs):
            for jj, fl_ipt in enumerate(fl.inputs):
                if fl_ipt == ipt:
                    self.good_ipts[ii] = fl.good_ipts[jj]
                    # TODO: add some treatment for c_x2 (mean? test diff?)
                    self.postns[ii][0] = fl.c_x1[jj]
                    self.postns[ii][1] = fl.c_y[jj]
                    self.expostns[ii][0] = fl.expx[jj]
                    self.expostns[ii][1] = fl.expy[jj]

        good_frac = float(np.sum(fl.good_ipts)) / float(fl.good_ipts.size)
        return good_frac

    def get_res_sing_src(self, fl):
        """ """
        for ii, ipt in enumerate(self.inputs):
            for jj, fl_ipt in enumerate(fl.inputs):
                if fl_ipt == ipt:
                    self.good_ipts[ii] = fl.good_ipts[jj]

        good_frac = float(np.sum(fl.good_ipts)) / float(fl.good_ipts.size)
        return good_frac
