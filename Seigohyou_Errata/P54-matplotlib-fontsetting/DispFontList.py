import matplotlib.font_manager as fm

# Get the list of installed fonts
font_list = [f.name for f in fm.fontManager.ttflist]
#print(font_list)

# Display the fonts starting with 'A'
a_fonts = [font for font in font_list if font.startswith('A')]
print(a_fonts)

# Display the fonts starting with 'N'
n_fonts = [font for font in font_list if font.startswith('N')]
print(n_fonts)

# Display the fonts starting with 'P'
p_fonts = [font for font in font_list if font.startswith('P')]
print(p_fonts)


# Check if 'Arial' is in the list
if 'Arial' in font_list:
    print('Arial font found.')
else:
    print('Arial font not found.')

