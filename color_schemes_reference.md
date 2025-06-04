# Streamlit Color Schemes Reference

## Understanding Hex Colors
- Hex colors start with # followed by 6 characters (0-9, A-F)
- Format: #RRGGBB (Red, Green, Blue)
- Examples: #FF0000 (red), #00FF00 (green), #0000FF (blue)

## Current Color Scheme (Professional Blue)
```toml
primaryColor = "#1f77b4"           # Blue - for interactive elements
backgroundColor = "#ffffff"        # White - main content area
secondaryBackgroundColor = "#f0f2f6"  # Light gray - sidebar & widgets
textColor = "#262730"              # Dark gray - all text
```

## Alternative Color Schemes

### 1. Modern Purple
```toml
primaryColor = "#7C3AED"           # Vibrant purple
backgroundColor = "#FFFFFF"        # White
secondaryBackgroundColor = "#F9FAFB"  # Very light gray
textColor = "#1F2937"              # Dark gray
```

### 2. Professional Green
```toml
primaryColor = "#10B981"           # Emerald green
backgroundColor = "#FFFFFF"        # White
secondaryBackgroundColor = "#F0FDF4"  # Very light green tint
textColor = "#064E3B"              # Dark green
```

### 3. Corporate Red
```toml
primaryColor = "#DC2626"           # Professional red
backgroundColor = "#FFFFFF"        # White
secondaryBackgroundColor = "#FEF2F2"  # Very light red tint
textColor = "#1F2937"              # Dark gray
```

### 4. Dark Mode
```toml
primaryColor = "#FF6B6B"           # Coral red
backgroundColor = "#0E1117"        # Dark background
secondaryBackgroundColor = "#262730"  # Darker gray
textColor = "#FAFAFA"              # Light text
base = "dark"                      # Don't forget to change base theme
```

### 5. Ocean Theme
```toml
primaryColor = "#006994"           # Ocean blue
backgroundColor = "#FFFFFF"        # White
secondaryBackgroundColor = "#E6F3F7"  # Light blue tint
textColor = "#003049"              # Dark blue
```

### 6. Sunset Orange
```toml
primaryColor = "#F97316"           # Vibrant orange
backgroundColor = "#FFFBF5"        # Warm white
secondaryBackgroundColor = "#FFF7ED"  # Light orange tint
textColor = "#431407"              # Dark brown
```

### 7. Minimalist Gray
```toml
primaryColor = "#6B7280"           # Medium gray
backgroundColor = "#FFFFFF"        # White
secondaryBackgroundColor = "#F9FAFB"  # Light gray
textColor = "#111827"              # Almost black
```

### 8. Forest Green
```toml
primaryColor = "#059669"           # Forest green
backgroundColor = "#FFFFFF"        # White
secondaryBackgroundColor = "#ECFDF5"  # Light green
textColor = "#064E3B"              # Dark green
```

## How to Apply a Color Scheme

1. Open `.streamlit/config.toml`
2. Replace the color values with your chosen scheme
3. Save the file
4. Restart your Streamlit app
5. The new colors will be applied

## Color Tools

To create your own colors:
- Google "color picker" for an interactive tool
- Use https://coolors.co for color palette generation
- Try https://color.adobe.com for professional palettes

## Tips
- Keep good contrast between text and background
- Use lighter colors for backgrounds
- Ensure primaryColor stands out from backgrounds
- Test with both light and dark terminals 