"""
Creates a custom "one_light" template and adds it to pio.templates.
Only colors of background and zerolines are actually from one light colorscheme.
"""

import plotly.graph_objects as go
import plotly.io as pio

custom_template = go.layout.Template()

# Set the background colors
custom_template.layout.paper_bgcolor = "#fafafa"
custom_template.layout.plot_bgcolor = "#fafafa"

# Set the font for the title
custom_template.layout.title.font.family = "Serif"

custom_template.layout.hovermode = "closest"

# Customize axis
custom_template.layout.xaxis.showgrid = False
custom_template.layout.xaxis.showline = True
custom_template.layout.xaxis.zerolinecolor = "#eaeaea"
custom_template.layout.yaxis.showgrid = False
custom_template.layout.yaxis.showline = True
custom_template.layout.yaxis.zerolinecolor = "#eaeaea"

# Set default color for traces
custom_template.layout.colorway = [
    "lightseagreen", "lightsalmon", "steelblue", "lightpink", "plum",
    "skyblue", "darkseagreen", "darkgray", "darksalmon", "mediumturquoise", "lightcoral",
    "palegreen", "orchid", "powderblue", "thistle", "lightslategray",
    "peachpuff", "mistyrose", "lavender", "aquamarine", "wheat", "paleturquoise",
    "sandybrown", "lightcyan", "lightpink", "khaki",
    "mediumaquamarine", "lemonchiffon", "pink", "palevioletred",
    "moccasin", "burlywood", "gainsboro", "rosybrown", "palegoldenrod"]
custom_template.layout.colorscale = {"sequential": "purpor_r", "diverging": "Tealrose_r"}
# Burg_r

custom_template.layout.coloraxis.colorbar.len = 0.75
custom_template.layout.coloraxis.colorbar.thickness = 20

# Register the template
pio.templates["one_light"] = custom_template
