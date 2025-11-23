# Roller Pump Cerebral Model

A geometry-based **peristaltic roller pump model** coupled to a **lumped-parameter neonatal cerebrovascular model** with **dynamic cerebral autoregulation (CA)**.

This repository contains the full implementation used for synthetic simulations in the accompanying Master's thesis in Technical Medicine.  
The model combines:

- A **roller-pump flow generator** based purely on pump geometry and time-varying shaft speed,
- A **Windkessel-type circulatory model** of the neonatal aortic, systemic, cerebral, and venous compartments,
- A **myogenic-type cerebral autoregulation mechanism** regulating downstream cerebrovascular resistance,
- Demonstration notebooks that reproduce static and dynamic simulations.

The model is tailored to neonatal cardiopulmonary bypass (CPB).

---

## 1. Repository Structure

```text
roller-pump-cerebral-model/
├─ src/
│  └─ roller_pump_cerebral_model/
│       ├─ __init__.py
│       └─ PUMP_CV_CA_SIM.py       # roller pump + cerebrovascular model implementation
├─ notebooks/
│  ├─ Demo_synthetic_SIM.ipynb    # main demo / synthetic simulations
│  └─ Test.ipynb                  # simple import test for the model
├─ requirements.txt
├─ LICENSE
└─ README.md
```

The package can be imported as:

```python
from roller_pump_cerebral_model import RollerPumpCerebralModel
```

In the test notebook (`notebooks/Test.ipynb`) this can be tested via.

---

## 2. Model Overview

### 2.1 Roller Pump Compartment
The roller pump converts a time-varying shaft speed **RPM(t)** into an outlet-flow waveform  
via:

Q_out(t) = Q_nom(ω(t)) + ω(t) * Σ_k p(φ(t) - φ_k)


where:

- ω(t) = 2π/60 * RPM(t) is the angular speed,
- Q_nom is the nominal (conveyor) flow in the non-occluded tube segment,
- p(ψ) is the geometry-derived pulsatile displacement of each roller,
- Nr is the number of rollers,
- φ(t) is the instantaneous shaft angle.

The nominal flow is:

Q_nom(t) =
π * (d_i/2)^2 * ω(t) * (r_b – d_o/2) * n_occ / 1000

where n_occ ∈ [0,1] is the occlusion factor.
The pulsatile component is computed from tube geometry, indentation depth, and residual lumen area.

### 2.2 Circulatory Compartments

The cardiovascular and cerebrovascular systems are modelled using a **lumped-parameter Windkessel analogue**, including:

#### Aortic compartment
Contains:
- proximal compliance Ca1
- distal compliance Ca2
- aortic resistance Ra
- inertance La

These generate proximal (Va1) and distal (Va2) aortic pressures, where Va2 represents arterial blood pressure (ABP).

#### Systemic compartment
All non-cerebral systemic vessels are lumped into a single resistance Rs.
Because the systemic and cerebral beds are in parallel between Va2 and venous pressure Vv, their relative impedance determines the flow split.

#### Cerebral compartment
Represents the MCA and pial circulation:
- proximal resistance Rc1
- distal resistance Rc2(t) (time-varying via CA)
- compliance Cc
- inertance Lc

Flow through Rc1 corresponds to CBF, assuming constant MCA diameter.

#### Venous compartment
A compliant venous reservoir:
- compliance Cv
- drainage resistance R_drain

This produces a simulated central venous pressure Vv representing CVP.

### 2.3 Cerebral Autoregulation (CA)

CA adjusts the distal cerebrovascular resistance (pial) Rc2(t) in response to mean arterial pressure.

1. **Mean ABP** is computed via:

d(Va2_mean)/dt = ( Va2(t) – Va2_mean ) / tau_mean

2. **Target resistance**:

R_target(t) = r_set * [ 1 + S * ( Va2_mean – ABP_set ) ]

3. **Dynamic response**:

dRc2/dt = ( R_target – Rc2 ) / tau_r

4. **Physiological bounds**:

Rc2(t) ∈ [ r_set * f_min ,  r_set * f_max ]

Within this range CBF remains stable; outside it becomes pressure-passive.

---

## 3. Import model

### 3.1 Requirements

Minimal Python dependencies (see `requirements.txt`):

```text
numpy
scipy
matplotlib
```

### 3.2 Local import

use the `src/` directory directly in your notebooks:

```python
import sys
from pathlib import Path

project_root = Path().resolve().parents[0]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from roller_pump_cerebral_model import RollerPumpCerebralModel
```

---

## 4. Demo: Synthetic Simulations

Open the main demo notebook:

```text
notebooks/Demo_synthetic_SIM.ipynb
```

This notebook reproduces:

- **Static baseline simulation** at constant RPM  
  (preset representing an “average neonate undergoing CPB”),
- **Dynamic simulation 1: RPM ramp**  
  CA ON vs. OFF → pressure-passive vs. autoregulated CBF,
- **Dynamic simulation 2: Rs ramp**  
  Increase in systemic vascular resistance at constant pump speed,
- **Long slow RPM ramp**  
  to generate a Lassen-type autoregulation curve (CBF vs. ABP) with CA disabled and enabled.

Plots are shown inline.  
Saving to disk can be enabled by passing `save=True` to the plotting functions; figures are saved under:

```text
../Report_Figures
```

relative to the `notebooks/` folder.

The `notebooks/Test.ipynb` file is a minimal smoke test that verifies that the model can be imported and instantiated from `src/`.

---

## 5. Basic Usage

```python
from roller_pump_cerebral_model import RollerPumpCerebralModel

# Create model with 1 ms time step
model = RollerPumpCerebralModel(dt=1e-3)

# Load hemodynamic parameters for an "average neonate undergoing CPB"
model.load_preset("average_final")

# Define RPM schedule (constant pump speed)
t_cmd = [0.0, 60.0]
rpm_cmd = [52.0, 52.0]
model.set_rpm_schedule(t_cmd, rpm_cmd, mode="linear")

# Example parameter change:
# model.hemo.Ca_1 = 0.02  # [mL/mmHg]

# Run simulation:
t, sol, out = model.simulate(T=50.0, car_enable=True)
```

---

## 6. Thesis Reference

This code was developed and used as part of a Master's thesis in Technical Medicine.

You can add your thesis link and citation here, for example:

> **Thesis:**  
> *Insights Into Cerebral Perfusion Management in Neonates on Cardiopulmonary Bypass: 
a Computational Physiological Modeling Approach*  
> *Author:* M.G.L. van Veen  
> *Institution:* University of Twente  
> *Year:* 2025  
> *URL:* After it becomes available, a link to the thesis will be placed here.

---

## 7. Attribution

Parts of the roller-pump geometry model are adapted from the openly licensed MATLAB implementation:

**M.P. McIntyre (2019)**  
*Roller-pump-model-and-simulation*  
MIT License  
https://github.com/michaelmcintyre/roller-pump-model-and-simulation

The Python implementation:

- rewrites and adapts the geometric roller-pump formulation for ODE simulation,
- enables time-varying RPM input,
- integrates the pump with a neonatal cerebrovascular Windkessel model,
- adds dynamic cerebral autoregulation.

---

## 8. License

This project is released under the **MIT License**.  
See the `LICENSE` file for full terms and third-party attribution.
