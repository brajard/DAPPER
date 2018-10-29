# Utilities (non-math)

try:
  from common import *
except:
  from DAPPER.common import *


#########################################
# Progressbar
#########################################
def noobar(itrble, desc):
  """Simple progress bar. To be used if tqdm not installed."""
  L  = len(itrble)
  print('{}: {: >2d}'.format(desc,0), end='')
  for k,i in enumerate(itrble):
    yield i
    p = (k+1)/L
    e = '' if k<(L-1) else '\n'
    print('\b\b\b\b {: >2d}%'.format(int(100*p)), end=e)
    sys.stdout.flush()

# Get progbar description by inspecting caller function.
import inspect
def pdesc(desc):
  if desc is not None:
    return desc
  try:
    #stackoverflow.com/q/15608987
    DAC_name  = inspect.stack()[3].frame.f_locals['name_hook']
  except (KeyError, AttributeError):
    #stackoverflow.com/a/900404
    DAC_name  = inspect.stack()[2].function
  return DAC_name 

# Define progbar as tqdm or noobar
try:
  import tqdm
  def progbar(inds, desc=None, leave=1):
    if is_notebook:
      pb = tqdm.tqdm_notebook(inds,desc=pdesc(desc),leave=leave)
    else:
      pb = tqdm.tqdm(inds,desc=pdesc(desc),leave=leave,smoothing=0.3,dynamic_ncols=True)
    # Printing during the progbar loop (may occur with error printing)
    # can cause tqdm to freeze the entire execution. 
    # Seemingly, this is caused by their multiprocessing-safe stuff.
    # Disable this, as per github.com/tqdm/tqdm/issues/461#issuecomment-334343230
    try: pb.get_lock().locks = []
    except AttributeError: pass
    return pb
except ImportError as err:
  install_warn(err)
  def progbar(inds, desc=None, leave=1):
    return noobar(inds,desc=pdesc(desc))




#########################################
# Console input / output
#########################################

#stackoverflow.com/q/292095
import select
def poll_input():
  i,o,e = select.select([sys.stdin],[],[],0.0001)
  for s in i: # Only happens if <Enter> has been pressed
    if s == sys.stdin:
      return sys.stdin.readline()
  return None

# Can't get thread solution working (in combination with getch()):
# stackoverflow.com/a/25442391/38281

# stackoverflow.com/a/21659588/38281
# (Wait for) any key:
def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch
    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch
getch = _find_getch()



# Terminal color codes. Use:
termcolors={
    'blue'      : '\033[94m',
    'green'     : '\033[92m',
    'OKblue'    : '\033[94m',
    'OKgreen'   : '\033[92m',
    'WARNING'   : '\033[93m',
    'FAIL'      : '\033[91m',
    'ENDC'      : '\033[0m' ,
    'header'    : '\033[95m',
    'bold'      : '\033[1m' ,
    'underline' : '\033[4m' ,
}

def print_c(*args,color='blue',**kwargs):
  s = ' '.join([str(k) for k in args])
  print(termcolors[color] + s + termcolors['ENDC'],**kwargs)


import inspect
def spell_out(*args,sep=",   ",end='\n'):
  "Print (args), including its variable name."
  flocs = inspect.stack()[1].frame.f_locals # get caller's namespace

  for i,x in enumerate(args):
    sep_ = end if i==len(args)-1 else sep

    # Find all matches.
    # It's impossible to distinguish between them, all should be listed.
    ks = [k for k in flocs if flocs[k] is x]

    # Don't use sep.join() coz print() can hook into
    # ipython's pretty print, which is better than str().
    if   len(ks)==0: print(             x,         end=sep_)
    elif len(ks)==1: print(ks[0], ": ", x, sep="", end=sep_)
    elif len(ks)>=2: print(ks   , ": ", x, sep="", end=sep_)


def print_together(*args):
  "Print 1D arrays stacked together."
  print(np.vstack(args).T)


# Local np.set_printoptions. stackoverflow.com/a/2891805/38281
import contextlib
@contextlib.contextmanager
@functools.wraps(np.set_printoptions)
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

try:
  import tools.tabulate as tabulate_orig
except:
  import DAPPER.tools.tabulate as tabulate_orig
tabulate_orig.MIN_PADDING = 0
def tabulate(data,headr=(),formatters=(),inds='nice'):
  """
  Pre-processor for tabulate().
  Main task: transpose 'data' (list-of-lists).
  If 'data' is a dict, the 'headr' will be keys.
  'formatter': define formats to apply before relaying to pandas.
               Default: attr.__name__ (when applicable).
  Example:
  >>> print(tabulate(cfgs.distinct_attrs()))
  """

  # Extract dict
  if hasattr(data,'keys'):
    headr = list(data)
    data  = data.values()

  # Default formats
  if not formatters:
    formatters = ({
        'test'  : lambda x: hasattr(x,'__name__'),
        'format': lambda x: x.__name__
        },)
  # Apply formatting (if not applicable, data is just forwarded)
  for f in formatters:
    data = [[f['format'](j) for j in row] if f['test'](row[0]) else row for row in data]

  # Transpose
  data = list(map(list, zip(*data)))

  # Generate nice indices
  if inds=='nice':
    inds = ['[{}]'.format(d) for d in range(len(data))]
  else:
    pass # Should be True or False

  return tabulate_orig.tabulate(data,headr,showindex=inds)


def repr_type_and_name(thing):
  """Print thing's type [and name]"""
  s = "<" + type(thing).__name__ + '>'
  if hasattr(thing,'name'):
    s += ': ' + thing.name
  return s


class MLR_Print:
  """
  Multi-Line, Recursive repr (print) functionality.
  Set class variables to change look:
   - 'indent': indentation per level
   - 'ch': character to use for "spine" (e.g. '|' or ' ')
   - 'ordr_by_linenum': 0: alphabetically, 1: linenumbr, -1: reverse
  """
  indent=3
  ch='.'
  ordr_by_linenum = 0

  # numpy print options
  threshold=10
  precision=None

  # Recursion monitoring.
  _stack=[] # Reference using MLR_Print._stack, ...
  # not self._stack or self.__class__, which reference sub-class "instance".

  # Reference using self.excluded, to access sub-class "instance".
  excluded = [] # Don't include in printing
  excluded.append(re.compile('^_')) # "Private"
  excluded.append('name') # Treated separately

  included = []
  aliases  = {}

  def __repr__(self):
    with printoptions(threshold=self.threshold,precision=self.precision):
      # new line chars
      NL = '\n' + self.ch + ' '*(self.indent-1)

      # Infinite recursion prevention
      is_top_level = False
      if MLR_Print._stack == []:
        is_top_level = True
      if self in MLR_Print._stack:
        return "**Recursion**"
      MLR_Print._stack += [self]

      # Use included or filter-out excluded
      keys = self.included or filter_out(vars(self), *self.excluded)

      # Process attribute repr's
      txts = {}
      for key in keys:
        t = repr(getattr(self,key)) # sub-repr
        if '\n' in t:
          # Activate multi-line printing
          t = t.replace('\n',NL+' '*self.indent) # other lines
          t = NL+' '*self.indent + t             # first line
        t = NL + self.aliases.get(key,key) + ': ' + t # key-name
        txts[key] = t

      def sortr(x):
        if self.ordr_by_linenum:
          return self.ordr_by_linenum*txts[x].count('\n')
        else:
          return x.lower()

      # Assemble string
      s = repr_type_and_name(self)
      for key in sorted(txts, key=sortr):
        s += txts[key]

      # Empty _stack when top-level printing finished
      if is_top_level:
        MLR_Print._stack = []

      return s


class AlignedDict(OrderedDict):
  """Provide aligned-printing for dict."""
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
  def __str__(self):
    L = max([len(s) for s in self.keys()], default=0)
    s = " "
    for key in self.keys():
      s += key.rjust(L)+": "+repr(self[key])+"\n "
    s = s[:-2]
    return s
  def __repr__(self):
    return type(self).__name__ + "(**{\n " + str(self).replace("\n",",\n") + "\n})"
  def _repr_pretty_(self, p, cycle):
    # Is implemented by OrderedDict, so must overwrite.
    if cycle: p.text('{...}')
    else:     p.text(self.__repr__())
  
class Bunch(dict):
  def __init__(self,**kw):
    dict.__init__(self,kw)
    self.__dict__ = self


# From stackoverflow.com/q/22797580 and more
class NamedFunc():
  "Provides custom repr for functions."
  def __init__(self,_func,_repr):
    self._func = _func
    self._repr = _repr
    #functools.update_wrapper(self, _func)
  def __call__(self, *args, **kw):
    return self._func(*args, **kw)
  def __repr__(self):
    argnames = self._func.__code__.co_varnames[
      :self._func.__code__.co_argcount]
    argnames = "("+",".join(argnames)+")"
    return "<NamedFunc>"+argnames+": "+self._repr

class NameFunc():
  "Decorator version"
  def __init__(self,name):
     self.fn_name = name
  def __call__(self,fn):
      return NamedFunc(fn,self.fn_name)


#########################################
# Writing / Loading Independent experiments
#########################################

import glob
def get_numbering(glb):
  ls = glob.glob(glb+'*')
  return [int(re.search(glb+'([0-9]*).*',f).group(1)) for f in ls]

def rel_path(path,start=None,ext=False):
  path = os.path.relpath(path,start)
  if not ext:
    path = os.path.splitext(path)[0]
  return path

import socket
def save_dir(filepath,pre=''):
  """Make dir DAPPER/data/filepath_without_ext/hostname"""
  host = socket.gethostname().split('.')[0]
  path = rel_path(filepath)
  dirpath  = os.path.join(pre,'data',path,host,'')
  os.makedirs(dirpath, exist_ok=True)
  return dirpath

def prep_run(path,prefix):
  "Create data-dir, create (and reserve) path (with its prefix and RUN)"
  path  = save_dir(path)
  path += prefix+'_' if prefix else ''
  path += 'run'
  RUN   = str(1 + max(get_numbering(path),default=0))
  path += RUN
  print("Will save to",path+"...")
  subprocess.run(['touch',path]) # reserve filename
  return path, RUN


import subprocess
def distribute(script,sysargs,xticks,prefix='',nCore=0.99,xCost=None):
  """
  Parallelization.

  Runs 'script' either as master, worker, or stand-alone,
  depending on 'sysargs[2]'.

  Return corresponding
   - portion of 'xticks'
   - portion of 'rep_inds' (setting repeat indices)
   - save_path.

  xCost: The computational assumed: [(1-xCost) + xCost*S for S in xticks].
  This controls how the xticks array gets distributed to nodes. Example:
   - Set xCost to 0 for uniform distribution of xticks array.
   - Set xCost to 1 if the costs scale linearly with the setting.
  """

  # Make running count (rep_inds) of repeated xticks.
  # This is typically used to modify the experiment seeds.
  rep_inds = [ list(xticks[:i]).count(x) for i,x in enumerate(xticks) ]

  if len(sysargs)>2:
    if sysargs[2]=='PARALLELIZE':
      save_path, RUN = prep_run(script,prefix)

      # THIS SECTION CAN BE MODIFIED TO YOUR OWN QUEUE SYSTEM, etc.
      # ------------------------------------------------------------
      # The implemention here-below does not use any queing system,
      # but simply launches a bunch of processes.
      # It uses (gnu's) screen for coolness, coz then individual
      # worker progress and printouts can be accessed using 'screen -r'.

      # screenrc path. This config is the "master".
      rcdir = os.path.join('data','screenrc')
      os.makedirs(rcdir, exist_ok=True)
      screenrc  = os.path.join(rcdir,'tmp_screenrc_')
      screenrc += os.path.split(script)[1].split('.')[0] + '_run'+RUN

      HEADER = """
      # Auto-generated screenrc file for experiment parallelization.
      source $HOME/.screenrc
      screen -t bash bash # make one empty bash session
      """.replace('/',os.path.sep)
      import textwrap
      HEADER = textwrap.dedent(HEADER)
      # Other useful screens to launch
      #screen -t IPython ipython --no-banner # empty python session
      #screen -t TEST bash -c 'echo nThread $MKL_NUM_THREADS; exec bash'

      # Decide number of batches (i.e. processes) to split xticks into.
      from psutil import cpu_percent, cpu_count
      if isinstance(nCore,float): # interpret as ratio of total available CPU
        nBatch = round( nCore * (1 - cpu_percent()/100) * cpu_count() )
      else:       # interpret as number of cores
        nBatch = min(nCore, cpu_count())
      nBatch = min(nBatch, len(xticks))

      # Write workers to screenrc
      with open(screenrc,'w') as f:
        f.write(HEADER)
        for i in range(nBatch):
          iWorker = i + 1 # start indexing from 1
          f.write('screen -t W'+str(iWorker)+' ipython -i --no-banner '+
              ' '.join([script,sysargs[1],'WORKER',str(iWorker),str(nBatch),save_path])+'\n')
          # sysargs:      0        1         2            3        4            5
        f.write("")
      sleep(0.2)
      # Launch
      subprocess.run(['screen', '-dmS', 'run'+RUN,'-c', screenrc])
      print("Experiments launched. Use 'screen -r' to view their progress.")
      sys.exit(0)

    elif sysargs[2] == 'WORKER':
      iWorker   = int(sysargs[3])
      nBatch    = int(sysargs[4])

      # xCost defaults
      if xCost==None:
        if prefix=='N':
          xCost = 0.02
        elif prefix=='F':
          xCost = 0

      # Split xticks array to this "worker":
      if xCost==None:
        # Split uniformly
        xticks   = np.array_split(xticks,nBatch)[iWorker-1]
        rep_inds = np.array_split(rep_inds,nBatch)[iWorker-1]
      else:
        # Weigh xticks by costs, before splitting uniformly
        eps = 1e-6                       # small number
        cc  = (1-xCost) + xCost*xticks   # computational cost,...
        cc  = np.cumsum(cc)              # ...cumulatively
        cc /= cc[-1]                     # ...normalized
        # Find index dividors between cc such that cumsum deltas are approx const:
        divs     = [find_1st_ind(cc>c+1e-6) for c in linspace(0,1,nBatch+1)]
        divs[-1] = len(xticks)
        # Split
        xticks   = array(xticks)[divs[iWorker-1]:divs[iWorker]]
        rep_inds = array(rep_inds)[divs[iWorker-1]:divs[iWorker]]

      print("xticks partition index:",iWorker)
      print("=> xticks array:",xticks)

      # Append worker index to save_path
      save_path = sysargs[5] + '_W' + str(iWorker)
      print("Will save to",save_path+"...")
      
      # Enforcing individual core usage.
      # --------------------------------
      # Issue: numpy often tries to distribute calculations accross cores.
      #    This may yield some performance gain, but not much compared to
      #    experiment parallelization (as we do here), coz of overhead.
      # Solution: force numpy to only use a single core.
      # Unfortunately: it's platform-dependent
      #    => you might have to adapt the code to your platform.
      # Testing: set nBatch=1. Only a single core should be in use.
      #    Check using your system's process manager, e.g., 'top'.
      try: 
        import mkl
        mkl.set_num_threads(1) # For Mac with Anaconda
      except ImportError:
        os.environ["MKL_NUM_THREADS"] = "1" # For Linux with Anaconda
        # NB: This might not work. => Try to set it in your .bashrc instead.

    elif sysargs[2]=='EXPENDABLE' or sysargs[2]=='DISPOSABLE':
      save_path = os.path.join('data','expendable')

    else: raise ValueError('Could not interpret sys.args[1]')

  else:
    # No args => No parallelization
    save_path, _ = prep_run(script,prefix)

  return xticks, save_path, rep_inds



#########################################
# Multiprocessing
#########################################
import multiprocessing, signal

def multiproc_map(func,xx,**kwargs):
  """
  Multiprocessing.
  Basically a wrapper for multiprocessing.pool.map(). Deals with
   - additional, fixed arguments.
   - KeyboardInterruption (python bug?)

  See example use in mods/QG/core.py.
  """
  NPROC = multiprocessing.cpu_count()-1

  # stackoverflow.com/a/35134329/38281
  orig = signal.signal(signal.SIGINT, signal.SIG_IGN)
  pool = multiprocessing.Pool(NPROC)
  signal.signal(signal.SIGINT, orig)
  try:
    # stackoverflow.com/a/5443941/38281
    f   = functools.partial(func,**kwargs)
    res = pool.map_async(f, xx)
    res = res.get(60)
  except KeyboardInterrupt as e:
    # Not sure why, but something this hangs,
    # so we repeat the try-block to catch another interrupt.
    try:
      traceback.print_tb(e.__traceback__)
      pool.terminate()
      sys.exit(0)
    except KeyboardInterrupt as e2:
      traceback.print_tb(e2.__traceback__)
      pool.terminate()
      sys.exit(0)
  else:
    pool.close()
  pool.join()
  return res






#########################################
# Misc
#########################################

# # Temporarily set attribute values
# @contextlib.contextmanager
# def set_tmp(obj,attr,val):
#     tmp = getattr(obj,attr)
#     setattr(obj,attr,val)
#     yield 
#     setattr(obj,attr,tmp)

import contextlib
@contextlib.contextmanager
def set_tmp(obj, attr, val):
    """
    Temporarily set an attribute.
    code.activestate.com/recipes/577089
    """
    was_there = False
    tmp = None
    if hasattr(obj, attr):
      try:
        if attr in obj.__dict__:
          was_there = True
      except AttributeError:
        if attr in obj.__slots__:
          was_there = True
      if was_there:
        tmp = getattr(obj, attr)
    setattr(obj, attr, val)
    yield #was_there, tmp
    if not was_there: delattr(obj, attr)
    else:             setattr(obj, attr, tmp)



# Better than tic-toc !
import time
class Timer():
  """
  Usage:
  with Timer('<description>'):
    do_stuff()
  """
  def __init__(self, name=None):
      self.name = name

  def __enter__(self):
      self.tstart = time.time()

  def __exit__(self, type, value, traceback):
      #pass # Turn off timer messages
      if self.name:
          print('[%s]' % self.name, end='')
      print('Elapsed: %s' % (time.time() - self.tstart))

def find_1st(xx):
  try:                  return next(x for x in xx if x)
  except StopIteration: return None
def find_1st_ind(xx):
  try:                  return next(k for k in range(len(xx)) if xx[k])
  except StopIteration: return None

# stackoverflow.com/a/2669120
def sorted_human( lst ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(lst, key = alphanum_key)

def keep_order_unique(arr):
  "Undo the sorting that np.unique() does."
  _, inds = np.unique(arr,return_index=True)
  return arr[np.sort(inds)]

def filter_out(orig_list,*unwanted,INV=False):
  """
  Returns new list from orig_list with unwanted removed.
  Also supports re.search() by inputting a re.compile('patrn') among unwanted.
  """
  new = []
  for word in orig_list:
    for x in unwanted:
      try:
        # Regex compare
        rm = x.search(word)
      except AttributeError:
        # String compare
        rm = x==word
      if (not INV)==bool(rm):
        break
    else:
      new.append(word)
  return new

def all_but_1_is_None(*args):
  "Check if only 1 of the items in list are Truthy"
  return sum(x is not None for x in args) == 1

# From stackoverflow.com/q/3012421
class lazy_property(object):
    '''
    Lazy evaluation of property.
    Should represent non-mutable data,
    as it replaces itself.
    '''
    def __init__(self,fget):
      self.fget = fget
      self.func_name = fget.__name__

    def __get__(self,obj,cls):
      value = self.fget(obj)
      setattr(obj,self.func_name,value)
      return value



class AssimFailedError(RuntimeError):
    pass

def raise_AFE(msg,time_index=None):
  if time_index is not None:
    msg += "\n(k,kObs,fau) = " + str(time_index) + ". "
  raise AssimFailedError(msg)


def vectorize0(f):
  """
  Vectorize f for its 1st (index 0) argument.

  Compared to np.vectorize:
    - less powerful, but safer to only vectorize 1st argument
    - doesn't evaluate the 1st item twice
    - doesn't always return array

  Example:
  >>> @vectorize0
  >>> def add(x,y):
  >>>   return x+y
  >>> add(1,100)
  101
  >>> x = np.arange(6).reshape((3,-1))
  >>> add(x,100)
  array([[100, 101],
       [102, 103],
       [104, 105]])
  >>> add([20,x],100)
  [120, array([[100, 101],
        [102, 103],
        [104, 105]])]
  """
  @functools.wraps(f)
  def wrapped(x,*args,**kwargs):
    if hasattr(x,'__iter__'):
      out = [wrapped(xi,*args,**kwargs) for xi in x]
      if isinstance(x,np.ndarray):
        out = np.asarray(out)
    else:
      out = f(x,*args,**kwargs)
    return out
  return wrapped


