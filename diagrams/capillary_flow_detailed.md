```mermaid
flowchart TD
    %% Starting Point
    start[Raw Microscope Images]
    
    %% Processing Nodes
    moco[External: MOCO ImageJ Plugin]
    contrast[src/capillary_contrast.py]
    background[src/write_background_file.py]
    hasty[External: hasty.ai Segmentation]
    naming[src/name_capillaries.py]
    manual[Manual CSV Review/Edit]
    renaming[src/rename_capillaries.py]
    centerline[src/find_centerline.py]
    kymograph[src/make_kymograph.py]
    velocity[src/analysis/make_velocities.py]
    validation[scripts/gui_kymos.py]
    
    %% Data Nodes
    stabilizedImg[Stabilized Images]
    shiftsCSV[results.csv (Shift Data)]
    contrastImg[Contrast-Enhanced Images]
    bgImg[Background & StdDev Images]
    segMasks[Segmentation Masks]
    numCaps[Numbered Capillaries\n& CSV Files]
    updatedCSV[Updated CSV Files]
    namedCaps[Consistently Named\nCapillaries]
    capCenterlines[Capillary Centerlines\n& Radii]
    kymoImages[Kymograph Images]
    velData[Initial Velocity Data]
    valVelData[Validated Velocity Data]
    
    %% Connections
    start --> moco
    moco --> stabilizedImg
    moco --> shiftsCSV
    stabilizedImg --> contrast
    shiftsCSV --> background
    contrast --> contrastImg
    contrastImg --> background
    background --> bgImg
    bgImg --> hasty
    hasty --> segMasks
    segMasks --> naming
    naming --> numCaps
    numCaps --> manual
    manual --> updatedCSV
    updatedCSV --> renaming
    renaming --> namedCaps
    namedCaps --> centerline
    centerline --> capCenterlines
    capCenterlines --> kymograph
    stabilizedImg --> kymograph
    kymograph --> kymoImages
    kymoImages --> velocity
    velocity --> velData
    velData --> validation
    validation --> valVelData
    
    %% Styling
    classDef processNode fill:#f5f5f5,stroke:#333,stroke-width:2px
    classDef dataNode fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,stroke-dasharray: 5 5
    classDef manualNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef externalNode fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class contrast,background,naming,renaming,centerline,kymograph,velocity,validation processNode
    class start,stabilizedImg,shiftsCSV,contrastImg,bgImg,segMasks,numCaps,updatedCSV,namedCaps,capCenterlines,kymoImages,velData,valVelData dataNode
    class manual manualNode
    class moco,hasty externalNode
``` 