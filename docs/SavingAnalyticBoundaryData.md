# Saving Analytic Boundary Data to H5 File

## Overview

The `AnalyticTestCharacteristicExtract` executable can now save analytic boundary
data to an H5 file in the same format used by `CharacteristicExtract`. This
allows you to:

1. Generate reference boundary data from analytic solutions
2. Use the saved boundary data as input to a standard CharacteristicExtract run
3. Compare CCE results between analytic and numerical worldtube data

## Usage

To enable boundary data saving, add the `AnalyticBoundaryDataFilePrefix` option
to the `Cce` section of your input file:

```yaml
Cce:
  # Other options...
  ExtractionRadius: 30.0
  
  # Optional: Save boundary data to H5 file
  AnalyticBoundaryDataFilePrefix: "AnalyticBoundary"
  
  AnalyticSolution:
    BouncingBlackHole:
      # Analytic solution parameters...
```

If `AnalyticBoundaryDataFilePrefix` is set to a string value (e.g., 
`"AnalyticBoundary"`), the boundary data will be saved to a file named:
```
<prefix>CceR<radius>.h5
```

For example, with `ExtractionRadius: 30.0` and 
`AnalyticBoundaryDataFilePrefix: "AnalyticBoundary"`, the output file will be:
```
AnalyticBoundaryCceR0030.h5
```

To disable boundary data saving (the default behavior), set the option to `Auto`:
```yaml
AnalyticBoundaryDataFilePrefix: Auto
```

## Output Format

The saved H5 file contains the Bondi boundary data in modal form, including:
- Beta
- J (Bondi J)
- DrJ (radial derivative of J)
- H (Du J, where Du is the derivative with respect to Bondi u)
- Q (Bondi Q)
- U (Bondi U)
- W (Bondi W)
- R (Bondi R)
- DuR (u-derivative of R)

Each dataset contains the time and the real and imaginary parts of the spherical
harmonic modes up to the specified LMax.

## Using Saved Boundary Data with CharacteristicExtract

The saved H5 file can be used directly as input to `CharacteristicExtract`:

```yaml
Executable: CharacteristicExtract

Cce:
  BoundaryDataFilename: AnalyticBoundaryCceR0030.h5
  ExtractionRadius: Auto  # Will be read from the filename
  # Other options...
```

This allows you to run CCE using the analytic boundary data, which is useful for
validation and testing purposes.

## Example Input Files

See the following example input files:
- `tests/InputFiles/Cce/AnalyticTestBouncingBlackHoleWithBoundaryOutput.yaml` - 
  Example with boundary data output enabled
- `tests/InputFiles/Cce/AnalyticTestBouncingBlackHole.yaml` - Example with
  boundary data output disabled (Auto)
