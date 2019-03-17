# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.misc import derivative as dd
from scipy.optimize import newton
import warnings
import contextlib
import os
import sys

# Avoid lsoda warning from being printed...
def fileno(file_or_fd):
  fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
  if not isinstance(fd, int):
    raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
  return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

##################################################################################################
## Input Parameters!
##################################################################################################

def ComputeWaterRocketHeight(FillRatio,DryMass,BottleVolume,NumberOfBoosters,LaunchTubeLength):
  # Physical constants
  Cv         = 718.0                         # specific heat at constant volume (J/kg/K)
  Cp         = 1005.0                        # specific heat at constant pressure (J/kg/K)
  GAMMA      = Cp / Cv                       # specific heat ratio (dimensionless)
  R          = Cp - Cv                       # ideal gas cst for air
  Patm       = 1.0*101325                    # atmospheric pressure (Pa)
  RHOatm     = 1.225                         # atmospheric density (kg/m^3)
  RHOw       = 1000.0                        # water density (kg/m^3)
  g          = 9.81                          # acceleration of gravity (m/s^2)

  # # Volumes in terms of specified fill factor
  Vb         = BottleVolume                    # tank volume (m^3)
  f          = FillRatio                       # fill factor (V_w / V_tank)
  V0a        = Vb * (1.0 - f)                  # initial volume of air in tank (m^3)   (computed from fill factor)
  V0w        = Vb * f                          # initial volume of water in tank (m^3) (computed from fill factor)

  # Volumes in terms of specified volumes
  # Vb         = 0.004                         # bottle volume (m^3)
  # V0w        = 0.001                         # initial volume of water in tank (m^3) (using a specified value)
  # V0a        = Vb - V0w                      # initial volume of air in tank (m^3)   (using a specified value)
  # f          = V0w / Vb                      # fill factor (V_w / V_tank)

  # Initial values
  P0a          = 6.0*101325                    # air pressure inside tank (Pa), 1 atm = 101325 Pa (absolute!!)
  T0a          = 15.0 + 273.15                 # air temperature inside tank (Cº)
  RHO0a        = P0a / ( R * T0a )             # initial air density

  # Mass values
  M0w          = RHOw * V0w                    # initial mass of 'propellant' water
  M0a          = RHO0a * V0a
  SloshingMass = NumberOfBoosters * M0w
  if (SloshingMass < 0.5):
    SloshingMass = 0.5
  Mdry         = SloshingMass + DryMass

  # Nozzle Geometry
  Rb        = 0.044                         # bottle radius (m)
  Rn        = 0.011                         # nozzle radius (m)
  Ln        = 0.10                          # nozzle length from the bottle's constant section to the exit
  SHAPE     = 'sigmoid'

  # Launch Tube Parameters
  LTubeL       = LaunchTubeLength
  LTubeOutD    = 0.95*(2*Rn)
  LTubeInnD    = LTubeOutD - 2.0*0.001      # Assuming a 1 mm thickness for the launch tube
  LTubeOutA    = np.pi * (LTubeOutD / 2.0) ** 2.0
  LTubeInnA    = np.pi * (LTubeInnD / 2.0) ** 2.0
  LauncherGasV = 0.001 # m^3 (= 1L)

  # Drag Coefficient
  Cd = 0.1

  # Initial Rocket Mass
  RocketInitialMass = Mdry + M0w * NumberOfBoosters + M0a

  # Drag Force
  def fDrag(Un):
    return - 0.5 * RHOatm * Ab * Cd * Un**(2.0)

  # Radius of nozzle at x
  def r(x):
    if x > Ln:
      return Rb
    else:
      if ( SHAPE == 'linear' ):
        # Linear shape
        slope = (Rb - Rn) / Ln
        return slope * x + Rn
      elif ( SHAPE == 'quadratic' ):
        # Quadratic shape
        aux = Rn**(2.0) + (Rb**(2.0) - Rn**(2.0)) / Ln * x
        return np.sqrt(aux)
      elif ( SHAPE == 'sigmoid' ):
        # Sigmoid shape
        k = 100.0
        return Rn+(Rb-Rn)/(1.0+np.exp(-k*(x-Ln/2.0)))

  # Plotting Bottle Shape

  # x_test = np.linspace(0,1.2*Ln,2000)
  # radius = np.zeros(2000)
  #
  # for i in range(0,2000):
  #   radius[i] = r(x_test[i])
  #
  # plt.figure()
  # plt.plot(x_test,radius,'k',x_test,-radius,'k')
  # plt.ylabel('r(x)')
  # plt.xlabel('x')
  # plt.grid()
  # plt.show()

  # Area function
  def A(x):
    return np.pi * r(x)**(2.0)

  # Inverse of Area function
  def Ainv(x):
    return (np.pi * r(x)**(2.0))**(-1.0)

  # Nozzle Area
  An = A(0)
  Ab = np.pi * Rb**(2.0)

  # Water volume in bottle
  def Vw(x):
    aux = quad(A, 0, x)[0]
    return aux

  # Find initial water height in bottle such that M0w is expelled at the end
  def findH0(x):
       return (quad(A,0,x)[0] * RHOw - M0w)

  # Printing stuff to screen

  # print('-> Geometric Parameters')
  # if ( SHAPE == 'linear' ):
  #   print('  Nozzle shape            :', SHAPE ,'.')
  # elif ( SHAPE == 'quadratic' ):
  #   print('  Nozzle shape            :', SHAPE, '.')
  #
  # print('  Nozzle radius at exit   :', Rn * 100.0, ' cm.')
  # print('  Nozzle length           :', Ln * 100.0, ' cm.')
  #
  # print('')
  # print('-> Initial Conditions')
  # print('  Initial water height      : ', H0 * 100.0, ' cm. (numerical)')
  # print('  Initial water volume      : ', Vw(H0),' L. (numerical)')
  # print('  Initial air mass          : ', M0a,' kg.')
  # print('  Rocket dry mass           : ', Mdry,' kg. (input)')
  # print('  Rocket wet mass           : ', M0w,' kg. (input)')
  # print('  Rocket total initial mass : ', RocketInitialMass,' kg.')

  ##################################################################################################
  ## Phase 0: Launch-Tube Phase
  ##################################################################################################

  # Setting up numerical scheme parameters
  t0Phase0      = 0.0
  tMaxPhase0    = 2.0
  dtPhase0      = 1e-5
  tPhase0       = np.arange(t0Phase0, tMaxPhase0, dtPhase0)
  tOutPhase0    = 0

  Vinit         = LauncherGasV + Vb - V0w - LTubeL * (LTubeOutA - LTubeInnA)
  nPhase0       = len(tPhase0)

  # Retrieving other physical quantities of interest
  ThrustPhase0       = np.zeros(nPhase0)
  PaPhase0           = np.zeros(nPhase0)
  RHOaPhase0         = np.zeros(nPhase0)
  TaPhase0           = np.zeros(nPhase0)
  ROCKETMTotalPhase0 = np.zeros(nPhase0)
  ROCKETACCPhase0    = np.zeros(nPhase0)
  ROCKETVPhase0      = np.zeros(nPhase0)
  ROCKETYPhase0      = np.zeros(nPhase0)

  for i in range(1, nPhase0):
    if (i != 0):
      RHOaPhase0[i]       = RHO0a * Vinit / (Vinit + ROCKETYPhase0[i-1] * LTubeInnA)
    else:
      RHOaPhase0[i]       = RHO0a

    PaPhase0[i]           = P0a * ( RHOaPhase0[i] / RHO0a ) ** GAMMA
    TaPhase0[i]           = T0a * ( RHOaPhase0[i] / RHO0a ) ** ( GAMMA - 1.0 )
    ThrustPhase0[i]       = NumberOfBoosters * ( PaPhase0[i] - Patm ) * LTubeInnA
    ROCKETMTotalPhase0[i] = RocketInitialMass
    if (i != 0):
      ROCKETACCPhase0[i]  = ( ThrustPhase0[i] + fDrag(ROCKETVPhase0[i-1]) ) /  ROCKETMTotalPhase0[i] - g
      ROCKETVPhase0[i]    = ROCKETVPhase0[i-1] + ROCKETACCPhase0[i] * (tPhase0[i] - tPhase0[i-1])
      ROCKETYPhase0[i]    = ROCKETYPhase0[i-1] + ROCKETVPhase0[i] * (tPhase0[i] - tPhase0[i-1])
    else:
      ROCKETACCPhase0[i]  = ThrustPhase0[i] / ROCKETMTotalPhase0[i] - g

    if (ROCKETYPhase0[i] >= LTubeL):
      break
    tOutPhase0 = tOutPhase0 + 1

  ThrustPhase0       = ThrustPhase0[0:tOutPhase0]
  PaPhase0           = PaPhase0[0:tOutPhase0]
  RHOaPhase0         = RHOaPhase0[0:tOutPhase0]
  TaPhase0           = TaPhase0[0:tOutPhase0]
  ROCKETMTotalPhase0 = ROCKETMTotalPhase0[0:tOutPhase0]
  ROCKETACCPhase0    = ROCKETACCPhase0[0:tOutPhase0]
  ROCKETVPhase0      = ROCKETVPhase0[0:tOutPhase0]
  ROCKETYPhase0      = ROCKETYPhase0[0:tOutPhase0]

  Pa01       = PaPhase0[-1]
  Rhoa01     = RHOaPhase0[-1]
  ROCKETV01  = ROCKETVPhase0[-1]
  ROCKETY01  = ROCKETYPhase0[-1]

  ##################################################################################################
  ## Phase 1: Water-Impulse Phase
  ##################################################################################################

  H0 = newton(findH0 , 0.1, tol=1E-5, maxiter=200)

  # Thrust in Phase 1 as a function of the nozzle velocity Un
  def fThrustPhase1(Un):
    return RHOw * An * Un**(2.0)

  # Internal non stationary term in Phase 1
  def fIntPhase1(Un,H,dUndt):
    return - RHOw * An * ( H * dUndt + (An / A(H)) * Un**(2.0) )

  # B variable in Phase 1: Water-Impulse Phase
  def fBPhase1(x):
    aux = quad(Ainv, 0, x)[0]
    return An * aux

  # C variable in Phase 1: Water-Impulse Phase
  def fCPhase1(x):
    return 0.5 * ( ( An / A(x) )**(2.0) - 1.0 )

  # D variable in Phase 1: Water-Impulse Phase
  def fDPhase1(x):
    return ( ( Vb - V0w ) / ( Vb - Vw(x) ) )**(GAMMA)

  # Air density inside the bottle in Phase 1
  def fRHOaPhase1(H):
    return Rhoa01 * ( fDPhase1(H) )**(1.0 / GAMMA)

  # Total Mass of Rocket during Phase 1
  def fMTotalPhase1(H):
    return Mdry + NumberOfBoosters*Vw(H) * ( RHOw - fRHOaPhase1(H) ) + NumberOfBoosters*fRHOaPhase1(H) * Vb

  # print('')
  # print('-> Phase 1: Water-Impulse Phase')
  # print('   Solving system of ODEs ...')

  # Defining System of ODEs
  def fPhase1(X, t):
    H     = X[0]
    Un    = X[1]
    dHdt  = ( An / A(H) ) * Un
    dUndt = 1.0 / ( fBPhase1(H) - RHOw * An * H**(2.0) / fMTotalPhase1(H) ) * (
            Patm / RHOw - fDPhase1(H) * Pa01 / RHOw - fDrag(Un) / fMTotalPhase1(H) -
            Un**(2.0) * ( fCPhase1(H) + H * RHOw * An / fMTotalPhase1(H) * ( 1.0 - An / A(H) ) ) )
    return [dHdt, dUndt]

  # Setting up numerical scheme parameters
  t0Phase1   = 0.0
  H0Phase1   = H0
  Un0Phase1  = 0.0
  X0Phase1   = [H0Phase1, Un0Phase1]
  tMaxPhase1 = 4.0
  dtPhase1   = 1e-5
  tPhase1    = np.arange(t0Phase1, tMaxPhase1, dtPhase1)

  # Filtering warning relating to excess work done due to the fact that when the water-air boundary crosses 0 it becomes
  # non-physical and weird stuff occurs
  warnings.filterwarnings(
      action='ignore',
      category=Warning,
      module=r'.*odepack'
  )

  # Solving system of ODEs
  with stdout_redirected():
    solPhase1  = odeint(fPhase1, X0Phase1, tPhase1)

  # print('   Successfully solved system of ODEs.')

  # Retrieving solution
  HPhase1    = solPhase1[:, 0]
  UnPhase1   = solPhase1[:, 1]

  # Finding where the Water-Air boundary crosses the nozzle exit (end of Phase 1)
  tOutPhase1 = 0
  while (HPhase1[tOutPhase1] > 0.00001):
    tOutPhase1 = tOutPhase1 + 1

  # print('   Time required for water to leave the rocket: ', tPhase1[tOutPhase1], ' s.'

  # Retrieving other physical quantities of interest
  ThrustPhase1       = np.zeros(tOutPhase1)
  MassFlowPhase1     = np.zeros(tOutPhase1)
  PaPhase1           = np.zeros(tOutPhase1)
  RHOaPhase1         = np.zeros(tOutPhase1)
  TaPhase1           = np.zeros(tOutPhase1)
  ROCKETMTotalPhase1 = np.zeros(tOutPhase1)
  ROCKETACCPhase1    = np.zeros(tOutPhase1)
  ROCKETVPhase1      = np.zeros(tOutPhase1)
  ROCKETYPhase1      = np.zeros(tOutPhase1)
  dUndtPhase1        = np.zeros(tOutPhase1)

  # Defining initial velocity and position of rocket
  ROCKETVPhase1[0]   = ROCKETV01
  ROCKETYPhase1[0]   = ROCKETY01

  dUndtPhase1        = np.gradient(UnPhase1, dtPhase1, edge_order=2)

  for i in range(0, tOutPhase1):
    ThrustPhase1[i]       = RHOw * An * UnPhase1[i]**(2.0)
    MassFlowPhase1[i]     = RHOw * An * UnPhase1[i]
    RHOaPhase1[i]         = Rhoa01 * fDPhase1(HPhase1[i])**(1.0 / GAMMA)
    PaPhase1[i]           = Pa01 * ( (Vb - V0w) / (Vb - Vw(HPhase1[i])) )**(GAMMA)
    TaPhase1[i]           = PaPhase1[i] / ( R * RHOaPhase1[i] )
    ROCKETMTotalPhase1[i] = fMTotalPhase1(HPhase1[i])
    ROCKETACCPhase1[i]    = ( NumberOfBoosters*fThrustPhase1(UnPhase1[i]) + NumberOfBoosters*fIntPhase1(UnPhase1[i],HPhase1[i],dUndtPhase1[i]) + fDrag(ROCKETVPhase1[i-1]) ) / ROCKETMTotalPhase1[i] - g
    if ( i >= 1 ):
      ROCKETVPhase1[i]    = ROCKETVPhase1[i-1] + ROCKETACCPhase1[i] * ( tPhase1[i] - tPhase1[i-1] )
      ROCKETYPhase1[i]    = ROCKETYPhase1[i-1] + ROCKETVPhase1[i]   * ( tPhase1[i] - tPhase1[i-1] )

  # Finding variables of interest at the end of Phase 1
  Pa02           = PaPhase1[-1]
  RHOa02         = RHOaPhase1[-1]
  Ta02           = TaPhase1[-1]
  ROCKETMTot02   = ROCKETMTotalPhase1[-1]
  ROCKETThrust02 = ThrustPhase1[-1]
  ROCKETACC02    = ROCKETACCPhase1[-1]
  ROCKETV02      = ROCKETVPhase1[-1]
  ROCKETY02      = ROCKETYPhase1[-1]

  # print('')
  # print('   Mass of rocket after Phase 1                : ', ROCKETMTot02, ' kg.')
  # print('   Air Pressure inside bottle after Phase 1    : ', Pa02, ' Pa.')
  # print('   Air Temperature inside bottle after Phase 1 : ', Ta02 - 273.15, ' ºC.')
  # print('   Air Density inside bottle after Phase 1     : ', RHOa02, ' kg/m^3.')
  # print(''
  # print('   Rocket thrust after Phase 1                 : ', ROCKETThrust02, ' N.')
  # print('   Rocket acceleration after Phase 1           : ', ROCKETACC02, ' m/s^2.')
  # print('   Rocket velocity after Phase 1               : ', ROCKETV02, ' m/s.')
  # print('   Rocket height after Phase 1                 : ', ROCKETY02, ' m.')

  ##################################################################################################
  ## Phase 2: Air Blowdown Phase
  ##################################################################################################
  ## Regime 1: Choked Flow
  ##################################################################################################

  # print('')
  # print('-> Phase 2: Air Blowdown, Regime 1 - Choked Flow')
  # print('   Solving ODE ...')

  # Defining System of ODEs
  def fPhase2Reg1(X, t):
    RHOt   = X[0]
    dRHOdt = - An / Vb * ( (GAMMA + 1.0)/2.0 )**( 1.0/(1.0-GAMMA) - 0.5 ) * RHOt * ( R*GAMMA*Ta02 )**(0.5) * ( RHOt / RHOa02 )**((GAMMA - 1.0) / 2.0)
    return dRHOdt

  # Setting up numerical scheme parameters
  t0Phase2Reg1   = 0.0
  X0Phase2Reg1   = RHOa02
  tMaxPhase2Reg1 = 2.0
  dtPhase2Reg1   = 1e-5
  tPhase2Reg1    = np.arange(t0Phase2Reg1, tMaxPhase2Reg1, dtPhase2Reg1)
  nTSPhase2Reg1  = len(tPhase2Reg1)

  # Solving system of ODEs
  with stdout_redirected():
    solPhase2Reg1  = odeint(fPhase2Reg1, X0Phase2Reg1, tPhase2Reg1)

  # Retrieving solution
  RHOaPhase2Reg1 = solPhase2Reg1[:, 0]

  # print('   Successfully solved ODE.'

  # Computing tank pressure from density
  PaPhase2Reg1aux = np.zeros(nTSPhase2Reg1)

  for i in range(0,nTSPhase2Reg1):
    PaPhase2Reg1aux[i] = Pa02 * ( RHOaPhase2Reg1[i] / RHOa02 )**(GAMMA)

  tOutPhase2Reg1 = 0
  while (PaPhase2Reg1aux[tOutPhase2Reg1] >= Patm * ((GAMMA+1.0)/2.0)**(GAMMA / (GAMMA - 1.0))):
    tOutPhase2Reg1 = tOutPhase2Reg1 + 1

  if (tOutPhase2Reg1 != 0):

    # print('   Time spent in choked flow regime               : ', tPhase2Reg1[tOutPhase2Reg1], ' s.')

    # Retrieving other physical quantities of interest
    ThrustPhase2Reg1       = np.zeros(tOutPhase2Reg1)
    MassFlowPhase2Reg1     = np.zeros(tOutPhase2Reg1)
    PaPhase2Reg1           = np.zeros(tOutPhase2Reg1)
    TaPhase2Reg1           = np.zeros(tOutPhase2Reg1)
    PnPhase2Reg1           = np.zeros(tOutPhase2Reg1)
    RHOnPhase2Reg1         = np.zeros(tOutPhase2Reg1)
    TnPhase2Reg1           = np.zeros(tOutPhase2Reg1)
    UnPhase2Reg1           = np.zeros(tOutPhase2Reg1)
    MnPhase2Reg1           = np.zeros(tOutPhase2Reg1)
    ROCKETMTotalPhase2Reg1 = np.zeros(tOutPhase2Reg1)
    ROCKETACCPhase2Reg1    = np.zeros(tOutPhase2Reg1)
    ROCKETVPhase2Reg1      = np.zeros(tOutPhase2Reg1)
    ROCKETYPhase2Reg1      = np.zeros(tOutPhase2Reg1)

    # Defining initial velocity and position of rocket
    ROCKETVPhase2Reg1[0]   = ROCKETV02
    ROCKETYPhase2Reg1[0]   = ROCKETY02

    for i in range(0, tOutPhase2Reg1):
      PaPhase2Reg1[i]           = Pa02 * ( RHOaPhase2Reg1[i] / RHOa02 )**(GAMMA)
      TaPhase2Reg1[i]           = Ta02 * (RHOaPhase2Reg1[i] / RHOa02) ** (GAMMA - 1.0)
      RHOnPhase2Reg1[i]         = RHOaPhase2Reg1[i] * ( (GAMMA + 1.0) / 2.0 )**(1.0 / (1.0 - GAMMA))
      PnPhase2Reg1[i]           = PaPhase2Reg1[i] * ( RHOnPhase2Reg1[i] / RHOaPhase2Reg1[i] ) ** (GAMMA)
      TnPhase2Reg1[i]           = TaPhase2Reg1[i] * ( RHOnPhase2Reg1[i] / RHOaPhase2Reg1[i] ) ** (GAMMA - 1.0)
      UnPhase2Reg1[i]           = - ( GAMMA * R * TnPhase2Reg1[i] )**(0.5)
      MnPhase2Reg1[i]           = 1.0
      MassFlowPhase2Reg1[i]     = RHOnPhase2Reg1[i] * An * UnPhase2Reg1[i]
      ThrustPhase2Reg1[i]       = 2.0 * PaPhase2Reg1[i] * An * ( 2.0 / (GAMMA+1.0) )**(1.0 / (GAMMA-1.0)) - Patm * An

      if (i == 0):
        ROCKETMTotalPhase2Reg1[i] = ROCKETMTot02
      else:
        ROCKETMTotalPhase2Reg1[i] = ROCKETMTotalPhase2Reg1[i-1] + NumberOfBoosters*MassFlowPhase2Reg1[i] * (tPhase2Reg1[i] - tPhase2Reg1[i-1])

      ROCKETACCPhase2Reg1[i]    = ( NumberOfBoosters*ThrustPhase2Reg1[i] + fDrag(ROCKETVPhase2Reg1[i-1]) ) / ROCKETMTotalPhase2Reg1[i] - g
      if (i >= 1):
        ROCKETVPhase2Reg1[i]      = ROCKETVPhase2Reg1[i-1] + ROCKETACCPhase2Reg1[i] * ( tPhase2Reg1[i] - tPhase2Reg1[i-1] )
        ROCKETYPhase2Reg1[i]      = ROCKETYPhase2Reg1[i-1] + ROCKETVPhase2Reg1[i]   * ( tPhase2Reg1[i] - tPhase2Reg1[i-1] )

    ROCKETMTot03   = ROCKETMTotalPhase2Reg1[-1]
    MassFlow03     = MassFlowPhase2Reg1[-1]
    Pa03           = PaPhase2Reg1[-1]
    Ta03           = TaPhase2Reg1[-1]
    RHOa03         = RHOaPhase2Reg1[-1]
    ROCKETThrust03 = ThrustPhase2Reg1[-1]
    ROCKETACC03    = ROCKETACCPhase2Reg1[-1]
    ROCKETV03      = ROCKETVPhase2Reg1[-1]
    ROCKETY03      = ROCKETYPhase2Reg1[-1]

    MassAirExpelledPhase2Reg1 = - NumberOfBoosters*np.trapz(MassFlowPhase2Reg1,x=tPhase2Reg1[0:tOutPhase2Reg1])

    # print('')
    # print('   Mass of air expelled during choked flow regime : ', MassAirExpelledPhase2Reg1, ' kg.')
    # print('   Mass of rocket after Phase 2.1                 : ', ROCKETMTot03, ' kg.')
    # print('   Air Pressure inside bottle after Phase 2.1     : ', Pa03, ' Pa.')
    # print('   Air Temperature inside bottle after Phase 2.1  : ', Ta03 - 273.15, ' ºC.')
    # print('   Air Density inside bottle after Phase 2.1      : ', RHOa03, ' kg/m^3.')
    # print(''
    # print('   Rocket thrust after Phase 2.1                  : ', ROCKETThrust03, ' N.')
    # print('   Rocket acceleration after Phase 2.1            : ', ROCKETACC03, ' m/s^2.')
    # print('   Rocket velocity after Phase 2.1                : ', ROCKETV03, ' m/s.')
    # print('   Rocket height after Phase 2.1                  : ', ROCKETY03, ' m.')

  else:

    # print('   Time spent in choked flow regime               : 0 s.')

    Pa03         = Pa02
    MassFlow03   = 0.0001
    ROCKETV03    = ROCKETV02
    ROCKETY03    = ROCKETY02
    ROCKETMTot03 = ROCKETMTot02
    MassAirExpelledPhase2Reg1 = 0.
    tOutPhase2Reg1 = 1
    MassFlowPhase2Reg1 = np.zeros(tOutPhase2Reg1)
    ROCKETMTotalPhase2Reg1    = np.zeros(tOutPhase2Reg1)
    ROCKETMTotalPhase2Reg1[0] = ROCKETMTot02
    ROCKETACCPhase2Reg1    = np.zeros(tOutPhase2Reg1)
    ROCKETVPhase2Reg1      = np.zeros(tOutPhase2Reg1)
    ROCKETYPhase2Reg1      = np.zeros(tOutPhase2Reg1)
    ThrustPhase2Reg1       = np.zeros(tOutPhase2Reg1)
    ROCKETACCPhase2Reg1[0] = ROCKETACCPhase1[-1]
    ROCKETVPhase2Reg1[0] = ROCKETVPhase1[-1]
    ROCKETYPhase2Reg1[0] = ROCKETYPhase1[-1]
    ThrustPhase2Reg1[0] = ThrustPhase1[-1]

  ##################################################################################################
  ## Regime 2: Subsonic Flow
  ##################################################################################################

  # print('')
  # print('-> Phase 2: Air Blowdown, Regime 2 - Subsonic Flow')
  # print('   Solving ODE ...')

  # Defining Auxiliary Functions
  def PratioTerm(P):
    return (1.0 - ( Patm / P )**((GAMMA - 1.0) / (GAMMA)))

  def eta(P):
    return np.sqrt( 2.0 * GAMMA / ( GAMMA - 1.0 ) * ( Patm / P )**(2.0 / GAMMA) * PratioTerm(P) )

  def pSqetaSq(P):
    return P**(2.0) * 2.0 * GAMMA / (GAMMA - 1.0) * (Patm / P)**(2.0 / GAMMA) * PratioTerm(P)

  # Defining System of ODEs
  def fPhase2Reg2(X, t):
    P       = X[0]
    Mdot    = X[1]
    dPdt    = - 2.0 * An**(2.0) * GAMMA**(2.0) / ( Vb * abs(Mdot) * (GAMMA - 1.0) ) * ( P**(2.0) * ( Patm / P )**((GAMMA + 1.0) / (GAMMA)) + Patm**(2.0) * ( Patm / P )**(-4.0*(GAMMA - 1.0) / (GAMMA)) * PratioTerm(P) ) * PratioTerm(P)
    if ((P > 1.00001*Patm or P < 0.999999*Patm) and Mdot != 0.0):
      dMdotdt = - abs(Mdot) / (2.0 * P) * ( ( ((GAMMA - 1.0) / GAMMA) * (2.0 + 1.0 / PratioTerm(P) * (Patm / P)**((GAMMA - 1.0) / (GAMMA)) ) ) * dPdt - An**(2.0) * pSqetaSq(P) / (Vb * abs(Mdot))   )
    elif (Mdot == 0.0):
      dMdotdt = 1.0 / (2.0 * P) * ( An**(2.0) * pSqetaSq(P) / Vb )
    else:
      dMdotdt = 0.0
    return [dPdt, dMdotdt]

  # Setting up numerical scheme parameters
  t0Phase2Reg2   = 0.0
  X0Phase2Reg2   = [Pa03,MassFlow03]
  tMaxPhase2Reg2 = 10.0
  dtPhase2Reg2   = 1e-5
  tPhase2Reg2    = np.arange(t0Phase2Reg2, tMaxPhase2Reg2, dtPhase2Reg2)

  # Solving System of ODEs
  with stdout_redirected():
    solPhase2Reg2         = odeint(fPhase2Reg2, X0Phase2Reg2, tPhase2Reg2)

  PaPhase2Reg2aux       = solPhase2Reg2[:,0]
  MassFlowPhase2Reg2aux = solPhase2Reg2[:,1]

  # print('   Successfully solved system of ODEs.')

  tOutPhase2Reg2 = 0
  while (PaPhase2Reg2aux[tOutPhase2Reg2] > 1.000001*Patm ):
    tOutPhase2Reg2 = tOutPhase2Reg2 + 1

  # print('   Time spent in Phase 2, subsonic regime         : ', tPhase2Reg2[tOutPhase2Reg2], ' s.')

  PaPhase2Reg2           = PaPhase2Reg2aux[0:tOutPhase2Reg2]
  MassFlowPhase2Reg2     = MassFlowPhase2Reg2aux[0:tOutPhase2Reg2]

  # Initializing remaining quantities of interest
  TaPhase2Reg2           = np.zeros(tOutPhase2Reg2)
  RHOaPhase2Reg2         = np.zeros(tOutPhase2Reg2)
  UnPhase2Reg2           = np.zeros(tOutPhase2Reg2)
  PnPhase2Reg2           = np.ones(tOutPhase2Reg2) * Patm
  TnPhase2Reg2           = np.zeros(tOutPhase2Reg2)
  RHOnPhase2Reg2         = np.zeros(tOutPhase2Reg2)
  MnPhase2Reg2           = np.zeros(tOutPhase2Reg2)
  ThrustPhase2Reg2       = np.zeros(tOutPhase2Reg2)
  ROCKETMTotalPhase2Reg2 = np.zeros(tOutPhase2Reg2)
  ROCKETACCPhase2Reg2    = np.zeros(tOutPhase2Reg2)
  ROCKETVPhase2Reg2      = np.zeros(tOutPhase2Reg2)
  ROCKETYPhase2Reg2      = np.zeros(tOutPhase2Reg2)

  # Defining initial velocity and position of rocket
  ROCKETVPhase2Reg2[0]   = ROCKETV03
  ROCKETYPhase2Reg2[0]   = ROCKETY03

  for i in range(0, tOutPhase2Reg2):
    TaPhase2Reg2[i]           = 1.0 / R * ( An * PaPhase2Reg2[i] / (-MassFlowPhase2Reg2[i]) )**(2.0) * eta(PaPhase2Reg2[i])**(2.0)
    RHOaPhase2Reg2[i]         = PaPhase2Reg2[i] / ( R * TaPhase2Reg2[i] )
    TnPhase2Reg2[i]           = TaPhase2Reg2[i] * (Patm / PaPhase2Reg2[i])**((GAMMA - 1.0) / GAMMA)
    RHOnPhase2Reg2[i]         = Patm / ( R * TnPhase2Reg2[i] )
    UnPhase2Reg2[i]           = MassFlowPhase2Reg2[i] / ( An * RHOnPhase2Reg2[i] )
    MnPhase2Reg2[i]           = - UnPhase2Reg2[i] / ( (GAMMA * R * TnPhase2Reg2[i]) ** 0.5 )
    ThrustPhase2Reg2[i]       = MassFlowPhase2Reg2[i] * UnPhase2Reg2[i]

    if (i == 0):
      ROCKETMTotalPhase2Reg2[i] = ROCKETMTot03
    else:
      ROCKETMTotalPhase2Reg2[i] = ROCKETMTotalPhase2Reg2[i-1] + NumberOfBoosters*MassFlowPhase2Reg2[i] * (tPhase2Reg2[i] - tPhase2Reg2[i-1])

    ROCKETACCPhase2Reg2[i]    = ( NumberOfBoosters*ThrustPhase2Reg2[i] + fDrag(ROCKETVPhase2Reg2[i-1]) ) / ROCKETMTotalPhase2Reg2[i] - g
    if (i >= 1):
      ROCKETVPhase2Reg2[i]      = ROCKETVPhase2Reg2[i-1] + ROCKETACCPhase2Reg2[i] * ( tPhase2Reg2[i] - tPhase2Reg2[i-1] )
      ROCKETYPhase2Reg2[i]      = ROCKETYPhase2Reg2[i-1] + ROCKETVPhase2Reg2[i]   * ( tPhase2Reg2[i] - tPhase2Reg2[i-1] )

  ROCKETMTot04   = ROCKETMTotalPhase2Reg2[-1]
  MassFlow04     = MassFlowPhase2Reg2[-1]
  Pa04           = PaPhase2Reg2[-1]
  Ta04           = TaPhase2Reg2[-1]
  RHOa04         = RHOaPhase2Reg2[-1]
  ROCKETThrust04 = ThrustPhase2Reg2[-1]
  ROCKETACC04    = ROCKETACCPhase2Reg2[-1]
  ROCKETV04      = ROCKETVPhase2Reg2[-1]
  ROCKETY04      = ROCKETYPhase2Reg2[-1]

  MassAirExpelledPhase2Reg2 = - NumberOfBoosters*np.trapz(MassFlowPhase2Reg2,x=tPhase2Reg2[0:tOutPhase2Reg2])
  TotalMassAirExpelled      = MassAirExpelledPhase2Reg1 + MassAirExpelledPhase2Reg2
  MassAirLeftInTank         = Vb*RHOa04

  # print('')
  # print('   Mass of air expelled during Phase 2.2          : ', MassAirExpelledPhase2Reg2, ' kg.')
  # print('   Mass of air left in tank after Phase 2.2       : ', MassAirLeftInTank, ' kg.')
  # print('   Air mass balance after Phase 2.2 (should be 0) : ', M0a - MassAirLeftInTank - TotalMassAirExpelled, ' kg.')
  # print('')
  # print('   Air Pressure inside bottle after Phase 2.2     : ', Pa04, ' Pa.')
  # print('   Air Temperature inside bottle after Phase 2.2  : ', Ta04 - 273.15, ' ºC.')
  # print('   Air Density inside bottle after Phase 2.2      : ', RHOa04, ' kg/m^3.')
  # print('')
  # print('   Rocket thrust after Phase 2.2                  : ', ROCKETThrust04, ' N.')
  # print('   Rocket acceleration after Phase 2.2            : ', ROCKETACC04, ' m/s^2.')
  # print('   Rocket velocity after Phase 2.2                : ', ROCKETV04, ' m/s.')
  # print('   Rocket height after Phase 2.2                  : ', ROCKETY04, ' m.')

  ##################################################################################################
  ## Phase 3: Free flight
  ##################################################################################################

  # print('')
  # print('-> Phase 3: Ballistic Flight')

  # Setting up numerical scheme parameters
  t0Phase3      = 0.0
  tMaxPhase3    = 20.0
  dtPhase3      = 1e-3
  tPhase3aux    = np.arange(t0Phase3, tMaxPhase3, dtPhase3)
  nTSPhase3aux  = len(tPhase3aux)

  # Initializing variables
  ROCKETMTotalPhase3aux = np.zeros(nTSPhase3aux)
  ROCKETACCPhase3aux    = np.zeros(nTSPhase3aux)
  ROCKETVPhase3aux      = np.zeros(nTSPhase3aux)
  ROCKETYPhase3aux      = np.zeros(nTSPhase3aux)

  # Defining initial velocity and position of rocket
  ROCKETVPhase3aux[0]   = ROCKETV04
  ROCKETYPhase3aux[0]   = ROCKETY04

  for i in range(0, nTSPhase3aux):

    ROCKETMTotalPhase3aux[i] = ROCKETMTot04
    if (i == 0):
      ROCKETACCPhase3aux[i]  = ( fDrag(ROCKETVPhase3aux[i])   ) / ROCKETMTotalPhase3aux[i] - g
    else:
      ROCKETACCPhase3aux[i]  = ( fDrag(ROCKETVPhase3aux[i-1]) ) / ROCKETMTotalPhase3aux[i] - g
    if (i >= 1):
      ROCKETVPhase3aux[i]    = ROCKETVPhase3aux[i-1] + ROCKETACCPhase3aux[i] * ( tPhase3aux[i] - tPhase3aux[i-1] )
      ROCKETYPhase3aux[i]    = ROCKETYPhase3aux[i-1] + ROCKETVPhase3aux[i]   * ( tPhase3aux[i] - tPhase3aux[i-1] )

  # Computing when the rocket reaches apogee height (maximum height)
  nTSPhase3 = np.argmax(ROCKETYPhase3aux) + 1

  # Initializing variables
  tPhase3            = tPhase3aux[0:nTSPhase3]
  ROCKETMTotalPhase3 = ROCKETMTotalPhase3aux[0:nTSPhase3]
  ROCKETACCPhase3    = ROCKETACCPhase3aux[0:nTSPhase3]
  ROCKETVPhase3      = ROCKETVPhase3aux[0:nTSPhase3]
  ROCKETYPhase3      = ROCKETYPhase3aux[0:nTSPhase3]

  ##################################################################################################
  ## Combining all variables into a single vector for all times
  ##################################################################################################

  # Combining all the times
  nTSTotal  = tOutPhase0 + tOutPhase1 + tOutPhase2Reg1 + tOutPhase2Reg2 + nTSPhase3
  tTotal    = np.concatenate(( np.concatenate(( np.concatenate(( np.concatenate(( tPhase0[0:tOutPhase0], tPhase0[tOutPhase0] + tPhase1[0:tOutPhase1]),axis=0 ),
                                                                 tPhase0[tOutPhase0] + tPhase1[tOutPhase1] + tPhase2Reg1[0:tOutPhase2Reg1]), axis=0 ),
                                                tPhase0[tOutPhase0] + tPhase1[tOutPhase1] + tPhase2Reg1[tOutPhase2Reg1] + tPhase2Reg2[0:tOutPhase2Reg2]),axis=0 ),
                               tPhase0[tOutPhase0] + tPhase1[tOutPhase1] + tPhase2Reg1[tOutPhase2Reg1] + tPhase2Reg2[tOutPhase2Reg2] + tPhase3), axis=0)

  # Initializing remaining total variables
  ROCKETThrustTotal  = np.zeros(nTSTotal)
  ROCKETMassTotal    = np.zeros(nTSTotal)
  ROCKETACCTotal     = np.zeros(nTSTotal)
  ROCKETVTotal       = np.zeros(nTSTotal)
  ROCKETYTotal       = np.zeros(nTSTotal)
  WaterMassFlowTotal = np.zeros(nTSTotal)
  AirMassFlowTotal   = np.zeros(nTSTotal)
  PaTotal            = np.zeros(nTSTotal)
  TaTotal            = np.zeros(nTSTotal)
  RHOaTotal          = np.zeros(nTSTotal)
  PanTotal           = np.zeros(nTSTotal)
  TanTotal           = np.zeros(nTSTotal)
  RHOanTotal         = np.zeros(nTSTotal)
  PwnTotal           = np.zeros(nTSTotal)
  TwnTotal           = np.zeros(nTSTotal)
  RHOwnTotal         = np.zeros(nTSTotal)
  ManTotal           = np.zeros(nTSTotal)
  UanTotal           = np.zeros(nTSTotal)
  UwnTotal           = np.zeros(nTSTotal)

  # Air and water mass flows
  WaterMassFlowTotal[tOutPhase0:tOutPhase0+tOutPhase1]                                                           = MassFlowPhase1[0:tOutPhase1]
  AirMassFlowTotal[tOutPhase0+tOutPhase1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1)]                                 = MassFlowPhase2Reg1[0:tOutPhase2Reg1]
  AirMassFlowTotal[tOutPhase0+tOutPhase1+tOutPhase2Reg1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2)-1] = MassFlowPhase2Reg2[1:tOutPhase2Reg2]

  # Rocket mass
  ROCKETMassTotal[0:tOutPhase0]                                                                                  = RocketInitialMass
  ROCKETMassTotal[tOutPhase0:tOutPhase0+tOutPhase1]                                                              = ROCKETMTotalPhase1[0:tOutPhase1]
  ROCKETMassTotal[tOutPhase0+tOutPhase1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1)]                                  = ROCKETMTotalPhase2Reg1[0:tOutPhase2Reg1]
  ROCKETMassTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1):(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2)]  = ROCKETMTotalPhase2Reg2[0:tOutPhase2Reg2]
  ROCKETMassTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2):nTSTotal]                                = ROCKETMTot04

  # Rocket Thrust
  ROCKETThrustTotal[0:tOutPhase0]                                                                                 = ThrustPhase0[0:tOutPhase0]
  ROCKETThrustTotal[tOutPhase0:tOutPhase0+tOutPhase1]                                                             = ThrustPhase1[0:tOutPhase1]
  ROCKETThrustTotal[tOutPhase0+tOutPhase1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1)]                                 = ThrustPhase2Reg1[0:tOutPhase2Reg1]
  ROCKETThrustTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1):(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2)] = ThrustPhase2Reg2[0:tOutPhase2Reg2]
  ROCKETThrustTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2):nTSTotal]                               = 0.0

  # Rocket Acceleration
  ROCKETACCTotal[0:tOutPhase0]                                                                                   = ROCKETACCPhase0[0:tOutPhase0]
  ROCKETACCTotal[tOutPhase0:tOutPhase0+tOutPhase1]                                                               = ROCKETACCPhase1[0:tOutPhase1]
  ROCKETACCTotal[tOutPhase0+tOutPhase1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1)]                                   = ROCKETACCPhase2Reg1[0:tOutPhase2Reg1]
  ROCKETACCTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1):(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2)]   = ROCKETACCPhase2Reg2[0:tOutPhase2Reg2]
  ROCKETACCTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2):nTSTotal]                                 = ROCKETACCPhase3[0:nTSPhase3]

  # Rocket Velocity
  ROCKETVTotal[0:tOutPhase0]                                                                                     = ROCKETVPhase0[0:tOutPhase0]
  ROCKETVTotal[tOutPhase0:tOutPhase0+tOutPhase1]                                                                 = ROCKETVPhase1[0:tOutPhase1]
  ROCKETVTotal[tOutPhase0+tOutPhase1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1)]                                     = ROCKETVPhase2Reg1[0:tOutPhase2Reg1]
  ROCKETVTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1):(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2)]     = ROCKETVPhase2Reg2[0:tOutPhase2Reg2]
  ROCKETVTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2):nTSTotal]                                   = ROCKETVPhase3[0:nTSPhase3]

  # Rocket Height
  ROCKETYTotal[0:tOutPhase0]                                                                                     = ROCKETYPhase0[0:tOutPhase0]
  ROCKETYTotal[tOutPhase0:tOutPhase0+tOutPhase1]                                                                 = ROCKETYPhase1[0:tOutPhase1]
  ROCKETYTotal[tOutPhase0+tOutPhase1:(tOutPhase0+tOutPhase1+tOutPhase2Reg1)]                                     = ROCKETYPhase2Reg1[0:tOutPhase2Reg1]
  ROCKETYTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1):(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2)]     = ROCKETYPhase2Reg2[0:tOutPhase2Reg2]
  ROCKETYTotal[(tOutPhase0+tOutPhase1+tOutPhase2Reg1+tOutPhase2Reg2):nTSTotal]                                   = ROCKETYPhase3[0:nTSPhase3]

  # print('-> Maximum Rocket Height : ', ROCKETYTotal[-1], ' m!')

  print('-> Vertical flight for ', tTotal[-1], ' seconds!')

  # #---------------- Plots, Plots, Plots ---------------#
  # ######################################################

  # # Total Final Plots
  # plt.figure()
  # plt.plot(tTotal[0:nTSTotal]*10**(3) ,  ROCKETThrustTotal[0:nTSTotal])
  # plt.ylabel('Rocket Thrust (N)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tTotal[0:nTSTotal]*10**(3) ,  ROCKETACCTotal[0:nTSTotal])
  # plt.ylabel('Rocket Acceleration (m/s^2)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tTotal[0:nTSTotal]*10**(3) ,  ROCKETVTotal[0:nTSTotal])
  # plt.ylabel('Rocket Velocity (m/s^2)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tTotal[0:nTSTotal]*10**(3) ,  ROCKETYTotal[0:nTSTotal])
  # plt.ylabel('Rocket Height (m)')
  # plt.xlabel('Time (ms)')
  # plt.grid()

  # Phase 1 Plots
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  HPhase1[0:tOutPhase1])
  # plt.ylabel('Position of Water-Air Boundary (m)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  UnPhase1[0:tOutPhase1])
  # plt.ylabel('Velocity of water through nozzle exit during Phase 1 (m/s)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  MassFlowPhase1[0:tOutPhase1])
  # plt.ylabel('Mass flow during Phase 1 (kg/s)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  ThrustPhase1[0:tOutPhase1])
  # plt.ylabel('Thrust during Phase 1 (N)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  PaPhase1[0:tOutPhase1])
  # plt.ylabel('Air Pressure during Phase 1 (Pa)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  RHOaPhase1[0:tOutPhase1])
  # plt.ylabel('Air Density during Phase 1 (kg/m3)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  TaPhase1[0:tOutPhase1])
  # plt.ylabel('Air Temperature during Phase 1 (K)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  ROCKETMTotalPhase1[0:tOutPhase1])
  # plt.ylabel('Rocket Mass during Phase 1 (Kg)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  ROCKETACCPhase1[0:tOutPhase1])
  # plt.ylabel('Rocket Acceleration during Phase 1 (m/s^2)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  ROCKETVPhase1[0:tOutPhase1])
  # plt.ylabel('Rocket Velocity during Phase 1 (m/s)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase1]*10**(3) ,  ROCKETYPhase1[0:tOutPhase1])
  # plt.ylabel('Rocket Height during Phase 1 (m)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # # Phase 2 plots
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  UnPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Velocity of water through nozzle exit during Phase 2 Regime 1 (m/s)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  MassFlowPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air mass flow through nozzle exit during Phase 2 Regime 1 (kg/s)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  PaPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air pressure in bottle during Phase 2 Regime 1 (Pa)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  TaPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air temperature in bottle during Phase 2 Regime 1 (K)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  RHOaPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air density in bottle during Phase 2 Regime 1 (kg/m^3)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  PnPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air pressure in nozzle during Phase 2 Regime 1 (Pa)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  TnPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air temperature in nozzle during Phase 2 Regime 1 (K)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  RHOnPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Air density in nozzle during Phase 2 Regime 1 (kg/m^3)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  ThrustPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Thrust during Phase 2 Regime 1 (N)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  ROCKETMTotalPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Rocket mass during Phase 2 Regime 1 (kg)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  ROCKETACCPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Rocket Acceleration during Phase 2 Regime 1 (m/s^2)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  ROCKETVPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Rocket velocity during Phase 2 Regime 1 (m/s)')
  # plt.xlabel('Time (ms)')
  # plt.grid()
  #
  # plt.figure()
  # plt.plot(tPhase1[0:tOutPhase2Reg1]*10**(3) ,  ROCKETYPhase2Reg1[0:tOutPhase2Reg1])
  # plt.ylabel('Rocket height during Phase 2 Regime 1 (m)')
  # plt.xlabel('Time (ms)')
  # plt.grid()

  # plt.show()

  return  RocketInitialMass, SloshingMass, ROCKETYTotal[-1]

StructuralMassString = input('Enter structural mass (no sloshing, Kg): ')
StructuralMass       = float(StructuralMassString)
BottleVolumeString   = input('Enter propellant bottle volume (L): ')
BottleVolume         = 0.001*float(BottleVolumeString)
NBoostersString      = input('Enter number of propellant bottles: ')
NBoosters            = int(NBoostersString)
LaunchTubeString     = input('Enter launch tube length (m): ')
LaunchTubeLength     = float(LaunchTubeString)
FillRatioLowerString = input('Enter lower limit of fill ratio (%): ')
FillRatioLower       = 0.01*float(FillRatioLowerString)
FillRatioUpperString = input('Enter upper limit of fill ratio (%): ')
FillRatioUpper       = 0.01*float(FillRatioUpperString)

# Running From Console
# DryMassNoSloshing = float(sys.argv[1])
# BottleVolume      = float(sys.argv[2])
# NBoosters         = int(sys.argv[3])

# Running from PyCharm
# StructuralMass    = 1.0
# BottleVolume      = 0.003
# NBoosters         = 1
# FillRatioLower    = 0.19
# FillRatioUpper    = 0.20
# LaunchTubeLength  = 0.5

if (FillRatioLower != FillRatioUpper):
  fillratios      = np.arange(FillRatioLower,FillRatioUpper,0.01)
  n               = len(fillratios)
else:
  fillratios      = np.ones(1)*FillRatioLower
  n               = 1

ROCKETY           = np.zeros(n)
InitialMass       = np.zeros(n)
SloshingMass      = np.zeros(n)
ResultsFolder     = 'ResultsNew'

print('\n')
print('Finding optimum fill ratio for a case with the following inputs:')
print('Structural Mass    = ', StructuralMass   ,' kg.')
print('Bottle Volume      = ', 1000*BottleVolume, ' L.')
print('Number of Boosters = ', NBoosters,           '.')
print('\n')

for i in range(0,n):
  InitialMass[i], SloshingMass[i], ROCKETY[i] = ComputeWaterRocketHeight(fillratios[i],StructuralMass,BottleVolume,NBoosters,LaunchTubeLength)
  print('Rocket height for a fill ratio of {fr:4.2f}%: {h:6.3f} m.'.format(fr=fillratios[i] * 100,h=ROCKETY[i]))

#np.savetxt("{}/RocketHeight_{m:4.2f}Kg_{bv:2d}L_{nb:2d}Boosters_{lt:3.1f}mLaunchTube.dat".format(ResultsFolder,m=StructuralMass,bv=int(BottleVolume*1000),nb=NBoosters,lt=LaunchTubeLength),np.transpose([fillratios,ROCKETY]),
#           header="Max Fill Ratio = {fr:3.1f}; Max Height = {mh:4.2f}; Sloshing Mass = {sm:3.1f}; Initial Total Mass = {im:3.1f}".format(fr=fillratios[np.argmax(ROCKETY)],mh=np.amax(ROCKETY),sm=SloshingMass[np.argmax(ROCKETY)],im=InitialMass[np.argmax(ROCKETY)]))

print('\n')
print('Maximum height of {mh:4.2f} m for a fill ratio of {fr:3.1f} % and an initial mass of {im:3.1f} kg.'.format(mh=np.amax(ROCKETY),fr=fillratios[np.argmax(ROCKETY)]*100.0,im=InitialMass[np.argmax(ROCKETY)]))

print('\n ----- End of Script ----- \n')
