unicode_string = "\\ud50c\\ub808\\uc774 \\ubd88\\ub2ed"
original_string = bytes(unicode_string, 'utf-8').decode('unicode-escape')

print(original_string)