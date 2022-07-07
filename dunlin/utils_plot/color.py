import seaborn           as sns

###############################################################################
#Globals
###############################################################################
#Refer for details: https://xkcd.com/color/rgb/
xkcd_colors = sns.colors.xkcd_rgb

def get_color(color):
    global colors
    return xkcd_colors.get(color, color)
    