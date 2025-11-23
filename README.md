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

\[
Q_\mathrm{out}(t) = Q_\mathrm{nom}(\omega(t)) + \omega(t) \sum_{k=1}^{N_r} p(\varphi(t)-\varphi_k),
\]

where:

- \( \omega(t) = 2\pi/60 \cdot \text{RPM}(t) \),
- \(Q_\mathrm{nom}\) is conveyor flow in a non-occluded segment,
- \(p(\psi)\) is the geometry-derived **pulsatile displacement** of a roller,
- \(N_r\) is the number of rollers,
- \(\varphi(t)\) is the shaft angle.

The nominal flow is:

\[
Q_\mathrm{nom}(t) =
\pi\left(\frac{d_i}{2}\right)^2 \omega(t)(r_b - d_o/2) \, n_{\text{occ}} / 1000,
\]

where \(n_\mathrm{occ} \in [0,1]\) is the occlusion factor.  
The pulsatile component is computed from tube geometry, indentation depth, and residual lumen area.

### 2.2 Circulatory Compartments

The cardiovascular and cerebrovascular systems are modelled using a **lumped-parameter Windkessel analogue**, including:

#### Aortic compartment
- Proximal compliance \(C_{a,1}\)
- Distal compliance \(C_{a,2}\)
- Aortic resistance \(R_a\)
- Inertance \(L_a\)

These generate proximal \(V_{a,1}\) and distal \(V_{a,2}\) pressures (the latter representing **ABP**).

#### Systemic compartment
All non-cerebral vascular beds lumped into a resistance \(R_s\).  
Because the systemic and cerebral branches are in parallel between \(V_{a,2}\) and venous pressure \(V_v\), their relative impedance determines the flow split.

#### Cerebral compartment
Represents the MCA and pial circulation:

- Proximal resistance \(R_{c,1}\)
- Distal resistance \(R_{c,2}(t)\)
- Compliance \(C_c\)
- Inertance \(L_c\)

Flow through \(R_{c,1}\) corresponds to **CBF** under the assumption of constant MCA diameter.

#### Venous compartment
A compliant CVP reservoir with:

- Compliance \(C_v\)
- Drainage resistance \(R_\text{drain}\)

### 2.3 Cerebral Autoregulation (CA)

CA adjusts the distal cerebrovascular resistance \(R_{c,2}(t)\) based on mean arterial pressure.

1. **Mean ABP** is computed via:

\[
\frac{d \overline{V_{a,2}}}{dt}
= \frac{V_{a,2}(t)-\overline{V_{a,2}}}{\tau_\text{mean}}.
\]

2. **Target resistance**:

\[
R_\text{target}(t)
= r_\text{set}\left[1 + S\big(\overline{V_{a,2}} - \mathrm{ABP}_\text{set}\big)\right].
\]

3. **Dynamic response**:

\[
\frac{dR_{c,2}}{dt}
= \frac{R_\text{target} - R_{c,2}}{\tau_r}.
\]

4. **Physiological bounds**:

\[
R_{c,2}(t)
\in [r_\text{set} f_\text{min},\; r_\text{set} f_\text{max}].
\]

Within this range CBF remains stable; outside it becomes pressure-passive.

---

## 3. Installation

### 3.1 Requirements

Minimal Python dependencies (see `requirements.txt`):

```text
numpy
scipy
matplotlib
```

### 3.2 Local installation

Option 1 – editable install from the project root:

```bash
pip install -e .
```

Option 2 – use the `src/` directory directly in your notebooks:

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
