import os, argparse, sys, logging, logging.handlers
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import poisson, itemfreq
from scipy.stats import ks_2samp
import scipy.special as special
from scipy.interpolate import interp1d, splrep, splev
from sklearn import cluster
from sklearn.neighbors import KDTree


## Customize the logger
log = logging.getLogger(__name__)


def load_TAD(path):
    """ Load TAD file.

    TAD file are tab splited file,
    which have at least 3 columns: ['chr', 'start', 'end']

    Parameters
    ----------
    path : str
        Path to TAD file.

    Return
    ------
    df : pandas.DataFrame
    """
    import pandas as pd
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.iloc[:, :3]
    df.columns = ['chr', 'start', 'end']
    df['chr'] = df['chr'].astype('str')
    if df.shape[0] > 0 and not df.iloc[0, 0].startswith('chr'):
        df['chr'] = 'chr' + df['chr']
    return df


def manipulation(matrix, start = 0):
    """Remove gaps of the original interaction matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray, (ndim = 2)
        Interaction matrix.
    
    start : int
        The begining of the region. (Default: 0)
    
    Returns
    -------
    newM : numpy.ndarray, (ndim = 2)
        The gap-removed matrix.
    
    convert : list
        The first element is the index map from *newM* to *matrix*, and
        the second element records the length of *matrix*.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mdknn import manipulation
    >>> matrix = np.random.rand(4, 4)
    >>> matrix[1,:] = 0; matrix[:,1] = 0
    >>> print matrix
    [[ 0.24822414  0.          0.07782508  0.01812965]
     [ 0.          0.          0.          0.        ]
     [ 0.93870151  0.          0.21986474  0.20462965]
     [ 0.13022712  0.          0.78674168  0.77068304]]
    
    >>> newM, convert = manipulation(matrix)
    >>> print newM
    [[ 0.24822414  0.07782508  0.01812965]
     [ 0.93870151  0.21986474  0.20462965]
     [ 0.13022712  0.78674168  0.77068304]]
    
    >>> print convert
    [{0: 0, 1: 2, 2: 3}, 4]

    """
    mask = matrix.sum(axis = 0) == 0
    index = list(np.where(mask)[0])
    # Remove vacant rows
    temp = np.delete(matrix, index, 0)
    # Vacant columns
    newM = np.delete(temp, index, 1)
    mapidx = dict(zip(np.arange(len(newM)),
                      np.where(np.logical_not(mask))[0] + start))
    convert = [mapidx, matrix.shape[0]]
    
    return newM, convert


def _fitting(x, y):
    
    ## Linear Spline
    ip = interp1d(x, y)
    # Downsample the data evenly
    times = np.arange(2, 4)
    scheck = x.size / times
    snum = scheck[scheck > 6][-1] if (scheck > 6).sum() > 0 else x.size
    snum = int(snum)
    xi = np.linspace(x.min(), x.max(), snum)
    yi = ip(xi)
    
    ## B-spline
    tcl = splrep(xi, yi)
    ys = splev(x, tcl)
    
    # Finite differences
    dy1 = np.gradient(ys)
    
    ## Unstable region
    m = (dy1[1:] >= 0) & (dy1[:-1] <= 0)
    if len(np.where(m)[0]) != 0:
        i = np.where(m)[0][0]
        ys[x > x[i]] = ys[i]
    
    return ys


class Core(object):
    """
    Interaction analysis at TAD level.
    
    High IFs off the diagonal region can be identified using
    :py:meth:`Core.longrange`. :py:meth:`Core.DBSCAN`
    performs a density-based clustering algorithm to detect aggregation patterns
    in those IFs. Furthermore, two structural features, called AP
    (Aggregation Preference) and Coverage in our original research, can be
    calculated by :py:meth:`Core.gdensity` and
    :py:meth:`Core.totalCover` respectively.
    
    Parameters
    ----------
    matrix : numpy.ndarray, (ndim = 2)
        Interaction matrix of a TAD.

    k : int, optional
        The number of nearest neighbors for calculate MDKNN. default 3.

    left : int, optional
        Starting point of TAD. For example, if the bin size is 10kb,
        ``left = 50`` means position 500000(bp) on the genome.
    
    Attributes
    ----------
    newM : numpy.ndarray, (ndim = 2)
        Gap-free interaction matrix.
    
    convert : list
        Information required for converting *newM* to *matrix*.
    
    cEM : numpy.ndarray, (ndim = 2)
        Expected interaction matrix. An upper triangular matrix. Value in each
        entry will be used to construct a Poisson Model for statistical
        significance calculation.
    
    fE : numpy.ndarray, (ndim = 2)
        An upper triangular matrix. Each entry represents the fold enrichment
        of corresponding observed interaction frequency.
    
    Ps : numpy.ndarray, (ndim = 2)
        An upper triangular matrix. Value in each entry indicates the p-value
        under corresponding Poisson Model.
    
    pos : numpy.ndarray, (shape = (N, 2))
        Coordinates of the selected IFs in *newM*.

    mean_dist_all : numpy.ndarray, (ndim = 1)
        Mean distance of center to k nearest neighbors of all pos.

    mean_dist : numpy.ndarray, (ndim = 1)
        Mean value of mean_dist_all.

    """
    def __init__(self, matrix, k=3, left = 0):
        matrix[np.isnan(matrix)] = 0
        self.k = k

        self.matrix = matrix

        # rescale matrix
        nonzero = matrix[matrix.nonzero()]
        if np.median(nonzero) < 1:
            min_nonzero = nonzero.min()
            scale = 1 / min_nonzero
            matrix = matrix * scale
        
        # Manipulation, remove vacant rows and columns
        self.newM, self.convert = manipulation(matrix, left)
        self._convert = np.array(list(self.convert[0].values()))
        
        ## Determine proper off-diagonal level
        Len = self.newM.shape[0]
        idiag = np.arange(0, Len)
        iIFs = []
        for i in idiag:
            temp = np.diagonal(self.newM, offset = i)
            iIFs.append(temp.mean())
        iIFs = np.array(iIFs)
        
        idx = np.where(iIFs > 0)[0][0]
        
        self._start = idx
        IFs = iIFs[idx:]
        diag = idiag[idx:]
        
        self._Ed = _fitting(diag, IFs)
        
    def longrange(self, pw = 2, ww = 5, top = 0.7, ratio = 0.05):
        """
        Select statistically significant interactions of the TAD. Both
        genomic distance and local interaction background are taken into
        account.
        
        Parameters
        ----------
        pw : int
            Width of the peak region. Default: 2
        
        ww : int
            Width of the donut. Default: 5
        
        top : float, [0.5, 1]
            Parameter for noisy interaction filtering. Default: 0.7
        
        ratio : float, [0.01, 0.1]
            Specifies the sample size of significant interactions.
            Default: 0.05
        
        Notes
        -----
        *pw* and *ww* are sensitive to data resolution. It performs well
        when we set *pw* to 4 and *ww* to 7 at 5 kb, and (2, 5) at 10 kb. [1]_
        
        References
        ----------
        .. [1] Rao, S.S., Huntley, M.H., Durand, N.C. et al. A 3D map of the
           human genome at kilobase resolution reveals principles of chromatin
           looping. Cell, 2014, 159: 1665-1680.
        
        """
        dim = self.newM.shape[0]
        
        ps = 2 * pw + 1 # Peak Size
        ws = 2 * ww + 1 # Initial window size
        bs = 2 * pw + 1 # B -- Blurry
        
        start = ww if (ww > self._start) else self._start
        # Upper triangular matrix
        upper = np.triu(self.newM, k = start)
        bUpper = np.triu(self.newM, k = 0)
        
        # Expanded Matrix
        expM = np.zeros((dim + ww*2, dim + ww*2))
        expBM = np.zeros((dim + ww*2, dim + ww*2))
        expM[ww:-ww, ww:-ww] = upper
        expBM[ww:-ww, ww:-ww] = bUpper
        
        tm = np.all((expBM == 0), axis = 0)
        Mask = np.zeros((dim + ww*2, dim + ww*2), dtype = bool)
        Mask[:,tm] = True
        Mask[tm,:] = True
        expCM = np.ones_like(expM, dtype = int)
        expCM[Mask] = 0
        
        ## Expected matrix
        EM_idx = np.triu_indices(dim, k = start)
        EM_value = self._Ed[EM_idx[1] - EM_idx[0] - self._start]
        EM = np.zeros((dim, dim))
        EM[EM_idx] = EM_value
        ## Expanded Expected Matrix
        expEM = np.zeros((dim + ww*2, dim + ww*2))
        expEM[ww:-ww, ww:-ww] = EM
        
        ## Construct pool of matrices for speed
        # Window
        OPool_w = {}
        EPool_w = {}
        ss = range(ws)
        for i in ss:
            for j in ss:
                OPool_w[(i,j)] = expM[i:(dim+i), j:(dim+j)]
                EPool_w[(i,j)] = expEM[i:(dim+i), j:(dim+j)]
        # Peak
        OPool_p = {}
        EPool_p = {}
        ss = range(ww-pw, ps+ww-pw)
        for i in ss:
            for j in ss:
                OPool_p[(i,j)] = expM[i:(dim+i), j:(dim+j)]
                EPool_p[(i,j)] = expEM[i:(dim+i), j:(dim+j)]
        
        # For Blurry Matrix
        OPool_b = {}
        OPool_bc = {}
        ss = range(ww-pw, bs+ww-pw)
        for i in ss:
            for j in ss:
                OPool_b[(i,j)] = expBM[i:(dim+i), j:(dim+j)]
                OPool_bc[(i,j)] = expCM[i:(dim+i), j:(dim+j)]
        
        ## Background Strength  --> Background Ratio
        bS = np.zeros((dim, dim))
        bE = np.zeros((dim, dim))
        for w in OPool_w:
            if (w[0] != ww) and (w[1] != ww):
                bS += OPool_w[w]
                bE += EPool_w[w]
        for p in OPool_p:
            if (p[0] != ww) and (p[1] != ww):
                bS -= OPool_p[p]
                bE -= EPool_p[p]
                
        bE[bE==0] = 1
        bR = bS / bE
        
        ## Corrected Expected Matrix
        cEM = EM * bR
        self.cEM = cEM
        
        ## Contruct the Blurry Matrix
        BM = np.zeros((dim, dim))
        CM = np.zeros((dim, dim), dtype = int)
        
        for b in OPool_b:
            BM += OPool_b[b]
            CM += OPool_bc[b]
        
        mBM = np.zeros_like(BM)
        Mask = CM != 0
        mBM[Mask] = BM[Mask] / CM[Mask]
        
        ## Fold Enrichments
        self.fE = np.zeros_like(self.cEM)
        mask = self.cEM != 0
        self.fE[mask] = upper[mask] / self.cEM[mask]
        
        ## Poisson Models
        Poisses = poisson(cEM)
        Ps = 1 - Poisses.cdf(upper)
        self.Ps = Ps
        rmBM = mBM[EM_idx] # Raveled array
        # Only consider the top x%
        top_idx = np.argsort(rmBM)[np.int(np.floor((1-top)*rmBM.size)):]
        # The most significant ones
        rPs = Ps[EM_idx][top_idx]
        Rnan = np.logical_not(np.isnan(rPs)) # Remove any invalid entry
        RrPs = rPs[Rnan]
        Np = np.int(np.ceil(ratio/top*RrPs.size))
        sig_idx = np.argsort(RrPs)[:Np]
        if sig_idx.size > 0:
            self.pos = np.r_['1,2,0', EM_idx[0][top_idx][Rnan][sig_idx], EM_idx[1][top_idx][Rnan][sig_idx]]
            self._pos = self.pos
            self.pos = self.convertPos(self.pos)
        else:
            self.pos = np.array([])
            
        self.Np = len(self.pos)
        self._area = EM_idx[0].size

    def pos2range(self, tad_chrom, tad_start, tad_end, binsize):
        """
        Convert longrange pos to it's genome range.
        """
        import pandas as pd
        df = pd.DataFrame(self.pos, columns=['bin1_id', 'bin2_id'])
        def to_range(bin_ids):
            start = tad_start + binsize * bin_ids
            end = start + binsize
            range_ = tad_chrom + ":" + start.astype('str') + "-" + end.astype('str')
            return range_
        df['tad_range'] = "{}:{}-{}".format(tad_chrom, tad_start, tad_end)
        df['bin1_range'] = to_range(df['bin1_id'])
        df['bin2_range'] = to_range(df['bin2_id'])
        return df
        
    def convertMatrix(self, M):
        """
        Convert an internal gap-free matrix(e.g., newM, cEM, fE, and Ps)
        into a new matrix with the same shape as the original interaction
        matrix by using the recorded index map(see the *convert* attribute).
        """
        idx = sorted(self.convert[0].values())
        newM = np.zeros((self.convert[1], self.convert[1]), dtype=M.dtype)
        y,x = np.meshgrid(idx, idx)
        newM[x,y] = M
            
        return newM

    def convertPos(self, pos):
        """
        Convert the coordinate of the points in the gap-free matrix
        into the coordinate in the original matrix.
        """
        new_x = self._convert[pos[:, 0]]
        new_y = self._convert[pos[:, 1]]
        new_pos = np.c_[new_x, new_y]
        return new_pos

    def MDKNN(self):
        """Cauculate MDKNN(Mean Distance of k Nearest Neighbors) of selected interactions.
        KD Tree is used for speed up nearest neighbor searching.

        See Also
        --------
        sklearn.neighbors.KDTree : an implementation of KDTree.
        """
        k = self.k
        if self.Np < k + 5: # Lower bound for input
            self.mean_dist = np.nan
            self.mean_dist_all = np.nan
            self.AP = np.nan
            self.local_ap = np.nan
            return
        
        self._kdtree = KDTree(self.pos)
        dist, ind = self._kdtree.query(self.pos, k=k+1)
        self._DKNN = dist[:, 1:]
        self._KNN = ind[:, 1:]

        self.mean_dist_all = self._DKNN.mean(axis=1)
        self.mean_dist = self.mean_dist_all.mean()
        #self.DoD = 1 - (1 / self.mean_dist_all).mean()
        self.DoD = self.mean_dist


class Compare(object):
    """Compare 2 sample, calculate the p-value and the difference(sample2 - sample1) of MDKNN.
    Statistical test using the two-sided Kolmogorov-Smirnov Test on 2 samples.

    Parameters
    ----------
    core1 : `Core`
        Core of sample1.

    core2 : `Core`
        Core of sample2.

    """
    def __init__(self, core1, core2):
        self.core1 = core1
        self.core2 = core2

    def compare(self):
        """
        Perform two sample KS-Test calculate p-value.

        See Also
        --------
        scipy.stats.ks_2samp : an implementation of KS-Test
        """
        dist1 = self.core1.mean_dist_all
        dist2 = self.core2.mean_dist_all
        D, pvalue = ks_2samp(dist1, dist2)
        diff = dist2.mean() - dist1.mean()
        self.D = D
        self.pvalue = pvalue
        self.diff = diff


def getmatrix(inter, l_bin, r_bin):
    """Extract regional interaction data and place it into a matrix.
    
    Parameters
    ----------
    inter : numpy structured array
        Three fields are required, "bin1", "bin2" and "IF", data types of
        which are int, int and float, respectively.
    
    l_bin : int
        Left bin index of the region.
        
    r_bin : int
        Right bin index of the region.
        
    Returns
    -------
    inter_matrix : numpy.ndarray
        The value of each entry is the interaction frequency between
        corresponding two bins.
        
    """
    # Construct a matrix
    inter_matrix = np.zeros((r_bin - l_bin, r_bin - l_bin), dtype = float)
    # Extract the regional data
    mask = (inter['bin1'] >= l_bin) & (inter['bin1'] < r_bin) & \
           (inter['bin2'] >= l_bin) & (inter['bin2'] < r_bin)
    inter_extract = inter[mask]
    
    # Fill the matrix
    for i in inter_extract:
        # Off-diagonal parts
        if i['bin1'] != i['bin2']:
            inter_matrix[i['bin1'] - l_bin][i['bin2'] - l_bin] += i['IF']
            inter_matrix[i['bin2'] - l_bin][i['bin1'] - l_bin] += i['IF']
        else:
            # Diagonal part
            inter_matrix[i['bin1'] - l_bin][i['bin2'] - l_bin] += i['IF']
    
    return inter_matrix

def getargs():
    ## Construct an ArgumentParser object for command-line arguments
    parser = argparse.ArgumentParser(usage = '%(prog)s [options]\n\n'
                                     'MDKNN -- Mean Distance of k Nearest Neighbors',
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # Output
    parser.add_argument('-O', '--output',
                        help = 'Output file name.')

    ## Argument Groups
    group_1 = parser.add_argument_group(title = 'Relate to the input')
    group_1.add_argument('-p', '--path', default = '.',
                         help = 'Path to the cool URI')
    group_1.add_argument('-p2', '--path-2',
                         help = 'Path to the cool URI')
    group_1.add_argument('-t', '--tad-file', help = 'Path to the TAD file.')
    group_1.add_argument('--out-long-range', help = 'Output the position of long-range points.')

    ## About the algorithm
    group_3 = parser.add_argument_group(title='Feature calculation')
    group_3.add_argument('-k', type = int, default = 3, help = 'Number of nearest neighbors.')
    group_3.add_argument('--pw', type = int, default = 2, help = '''Width of the interaction
                         region surrounding the peak. According to experience, we set it
                         to 1 at 20 kb, 2 at 10 kb, and 4 at 5 kb.''')
    group_3.add_argument('--ww', type = int, default = 5, help = '''Width of the donut region
                         Set it to 3 at 20 kb, 5 at 10 kb, and 7 at 5 kb.''')
    group_3.add_argument('--top', type = float, default = 0.7, help = 'Parameter for noisy '
                         'interaction filtering. By default, 30 percent noisy interactions'
                         ' will be eliminated.')
    group_3.add_argument('--ratio', type = float, default = 0.05, help = 'Specifies the sample'
                         ' ratio of significant interactions for TAD.')
    group_3.add_argument('--gap', type = float, default = 0.2, help = 'Maximum gap ratio.')
    
    ## Parse the command-line arguments
    commands = sys.argv[1:]
    if not commands:
        commands.append('-h')
    args = parser.parse_args(commands)
    
    return args, commands


def get_gap_ratio(matrix):
    matrix[np.isnan(matrix)] = 0
    newM, _ = manipulation(matrix)
    if len(matrix) > 0:
        # Ratio of Gaps (Vacant Rows or Columns)
        gaps = 1 - len(newM) / len(matrix)
    else:
        gaps = 1.0
    return newM, gaps


## Pipeline
def pipe(args, logger):
    """The Main pipeline for MDKNN.
    """
    ## Logging for argument setting
    arglist = ['# ARGUMENT LIST:',
               '# output file name = {0}'.format(args.output),
               '# Hi-C path = {0}'.format(args.path),
               '# TAD source file = {0}'.format(args.tad_file),
               '# Peak window width = {0}'.format(args.pw),
               '# Donut width = {0}'.format(args.ww),
               '# Noise filtering ratio = {0}'.format((1 - args.top)),
               '# Significant interaction ratio = {0}'.format(args.ratio),
               '# Maximum gap ratio = {0}'.format(args.gap)]
    
    argtxt = '\n'.join(arglist)
    logger.info('\n' + argtxt)
             
    logger.info('Read Hi-C data ...')
    cool = cooler.Cooler(args.path)

    # Load External TAD File, Columns 0,1,2
    logger.info('Read external TAD data ...')

    TADs = load_TAD(args.tad_file)
    
    # Header
    header = ['ChromID', 'Start', 'End', 'DoD', 'Gap-Ratio']
    out_rows = []

    if args.out_long_range:
        # Output long range positions
        long_range_pos = []

    logger.info('Calculate feature for each TAD ...')
    for i, row in tqdm(TADs.iterrows(), total=TADs.shape[0]):
        chr_, start, end = row.chr, row.start, row.end

        # Interaction Matrix
        balance = 'weight' in cool.bins().keys()
        try:
            matrix = cool.matrix(balance=balance, sparse=False)\
                       .fetch((chr_, start, min(end, cool.chromsizes[chr_])))
        except KeyError:
            if chr_.startswith('chr'):
                chr_ = chr_[3:]
            else:
                chr_ = 'chr'+chr_
            matrix = cool.matrix(balance=balance, sparse=False)\
                       .fetch((chr_, start, min(end, cool.chromsizes[chr_])))

        newM, gaps = get_gap_ratio(matrix)
        if (gaps < args.gap) and (newM.shape[0] > (args.ww * 2 + 1)):
            core = Core(matrix, k=args.k)
            # Extract Long-Range Interactions
            core.longrange(pw=args.pw, ww=args.ww, top=args.top, ratio=args.ratio)
            # Feature
            core.MDKNN()

            # Line by Line
            if core.mean_dist_all is not np.nan:
                o_row = [chr_, start, end, core.DoD, gaps]
            else:
                o_row = [chr_, start, end, np.nan, gaps]

            if args.out_long_range:
                df = core.pos2range(chr_, start, end, cool.binsize)
                long_range_pos.append(df)
        else:
            # Bad Domain!
            o_row = [chr_, start, end, np.nan, gaps]
        out_rows.append(o_row)
    logger.info('Done!')
    out_df = pd.DataFrame(out_rows)
    out_df.columns = header

    if args.out_long_range:
        logger.info("Write long-range point positions to %s ...", args.out_long_range)
        long_range_pos_df = pd.concat(long_range_pos)
        long_range_pos_df.to_csv(args.out_long_range, sep="\t", index=False)

    logger.info('Write results to %s ...', args.output)
    out_df.to_csv(args.output, na_rep='NA', sep='\t', index=False)
    logger.info('Done!\n')


def pipe_compare(args, logger):
    ## Logging for argument setting
    arglist = ['# ARGUMENT LIST:',
               '# output file name = {0}'.format(args.output),
               '# Comparision mode: True',
               '# Hi-C path 1 = {0}'.format(args.path),
               '# Hi-C path 2 = {0}'.format(args.path_2),
               '# TAD source file = {0}'.format(args.tad_file),
               '# Peak window width = {0}'.format(args.pw),
               '# Donut width = {0}'.format(args.ww),
               '# Noise filtering ratio = {0}'.format((1 - args.top)),
               '# Significant interaction ratio = {0}'.format(args.ratio),
               '# Maximum gap ratio = {0}'.format(args.gap)]

    argtxt = '\n'.join(arglist)
    logger.info('\n' + argtxt)

    logger.info('Read Hi-C data ...')
    cool_1 = cooler.Cooler(args.path)
    cool_2 = cooler.Cooler(args.path_2)
    assert cool_1.binsize == cool_2.binsize, "cooler1 and cooler2's binsize must same."

    # Load External TAD File, Columns 0,1,2
    logger.info('Read external TAD data ...')
    TADs = load_TAD(args.tad_file)

    out_rows = []
    # Header
    header = ['ChromID', 'Start', 'End', 'DoD1', 'DoD2', 'D', 'p-value', 'padj', 'Gap-Ratio-1', 'Gap-Ratio-2']

    if args.out_long_range:
        # Output long range positions
        long_range_pos = []

    logger.info('Calculate feature for each TAD ...')
    for i, row in tqdm(TADs.iterrows(), total=TADs.shape[0]):
        chr_, start, end = row.chr, row.start, row.end

        # Interaction Matrix
        balance = 'weight' in cool_1.bins().keys()
        try:
            matrix_1 = cool_1.matrix(balance=balance,  sparse=False)\
                       .fetch((chr_, start, min(end, cool_1.chromsizes[chr_])))
        except KeyError:
            if chr_.startswith('chr'):
                chr_ = chr_[3:]
            else:
                chr_ = 'chr'+chr_
            matrix_1 = cool_1.matrix(balance=balance,  sparse=False)\
                       .fetch((chr_, start, min(end, cool_1.chromsizes[chr_])))
        balance = 'weight' in cool_2.bins().keys()
        matrix_2 = cool_2.matrix(balance=balance,  sparse=False)\
                   .fetch((chr_, start, min(end, cool_2.chromsizes[chr_])))

        newM_1, gaps_1 = get_gap_ratio(matrix_1)
        newM_2, gaps_2 = get_gap_ratio(matrix_2)
            
        if (gaps_1 < args.gap) and (newM_1.shape[0] > (args.ww * 2 + 1)) and \
           (gaps_2 < args.gap) and (newM_2.shape[0] > (args.ww * 2 + 1)):

            core_1 = Core(matrix_1, k=args.k)
            core_1.longrange(pw = args.pw, ww = args.ww, top = args.top, ratio = args.ratio)
            core_1.MDKNN()

            core_2 = Core(matrix_2, k=args.k)
            core_2.longrange(pw = args.pw, ww = args.ww, top = args.top, ratio = args.ratio)
            core_2.MDKNN()
            comp = Compare(core_1, core_2)

            if args.out_long_range:
                df1 = core_1.pos2range(chr_, start, end, cool_1.binsize)
                df2 = core_2.pos2range(chr_, start, end, cool_2.binsize)
                tad_ = df1.pop('tad_range'); df2.pop('tad_range')
                df1.add_prefix('sample1_'); df2.add_prefix('sample2_')
                df = pd.concat([df1, df2, tad_], axis=1)
                long_range_pos.append(df)

            if (core_1.mean_dist_all is np.nan) or (core_2.mean_dist_all is np.nan):
                # Bad Domain!
                o_row = [chr_, start, end, np.nan, np.nan, np.nan, np.nan, np.nan, gaps_1, gaps_2]
            else:
                comp.compare()
                pvalue, D = comp.pvalue, comp.D
                o_row = [chr_, start, end, core_1.DoD, core_2.DoD, D, pvalue, np.nan, gaps_1, gaps_2]
        else:
            # Bad Domain!
            o_row = [chr_, start, end, np.nan, np.nan, np.nan, np.nan, np.nan, gaps_1, gaps_2]
            
        out_rows.append(o_row)
    
    out_df = pd.DataFrame(out_rows)
    out_df.columns = header

    logger.info("Perform multiple test correction with fdr_bh method.")
    pvals = out_df['p-value']
    pvals_ = pvals[~pvals.isna()]
    _, padjs, _, _ = multipletests(pvals_, method='fdr_bh')
    out_df.loc[pvals_.index, 'padj'] = padjs

    logger.info('Done!')
    logger.info('Write results to %s ...', args.output)
    out_df.to_csv(args.output, sep='\t', na_rep='NA', index=False)

    if args.out_long_range:
        logger.info("Write long-range point positions to %s ...", args.out_long_range)
        long_range_pos_df = pd.concat(long_range_pos)
        long_range_pos_df.to_csv(args.out_long_range, sep="\t", index=False)

    logger.info('Done!\n')


def main():
    # Parse Arguments
    args, commands = getargs()
    # Improve the performance if you don't want to run it
    if commands[0] not in ['-h', '-v', '--help', '--version']:
        ## Root Logger Configuration
        logger = logging.getLogger()
        logger.setLevel(10)
        console = logging.StreamHandler()
        filehandler = logging.handlers.RotatingFileHandler('mdknn.log',
                                                           maxBytes = 30000,
                                                           backupCount = 5)
        # Set level for Handlers
        console.setLevel('INFO')
        filehandler.setLevel('DEBUG')
        # Customizing Formatter
        formatter = logging.Formatter(fmt = '%(name)-14s %(levelname)-7s @ %(asctime)s: %(message)s',
                                      datefmt = '%m/%d/%y %H:%M:%S')
        ## Unified Formatter
        console.setFormatter(formatter)
        filehandler.setFormatter(formatter)
        # Add Handlers
        logger.addHandler(console)
        logger.addHandler(filehandler)

        # compare 2 sample or not
        comparision = False
        if args.path_2:
            comparision = True

        if comparision:
            pipe_compare(args, logger)
        else:
            pipe(args, logger)


if __name__ == '__main__':
    main()