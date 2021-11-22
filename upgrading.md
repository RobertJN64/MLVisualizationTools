Upgrading from 0.3.0 to 0.4.0.

This is a major updated intended to move forward to support beyond tensorflow
and new feature integration.

New Features:
 - graph() - autopicks graph type and mode
 - plotlyGraph() - autopicks graph mode
 - matplotlibGraph() - autopicks graph mode

Breaking Changes:
 - Major:
   - Analytics:
     - Tensorflow replaced by analyzeModel
     - TensorflowRaw replaced by analyzeModelRaw
     
   - Interfaces:
     - TensorflowGrid replace by predictionGrid
     - TensorflowAnimation replaced by pedictionAnimation
     - Corresponding raw replacements
     - 'Output' key is now customizable in interface calls
     
   - Colorizers:
     - Simple replaced by simple
     - Binary replaced by binary
     - 'Color' key is now customizable in colorizer calls
     
   - Graphs:
     - x, y, and anim params should no longer be passed to the graphs,
     they are autodetected instead

   - DashModelVisualizer
     - kagglenotebook param renamed to usetunneling
     
     

 - Minor:
   - Type enums moved into new file (types.py)