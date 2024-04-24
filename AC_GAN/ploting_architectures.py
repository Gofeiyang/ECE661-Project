import os
import subprocess

def create_latex_file(latex_content, filename="network_diagram.tex"):
    with open(filename, 'w') as file:
        file.write(latex_content)

def compile_latex_to_pdf(tex_file):
    # Compile LaTeX to PDF (make sure pdflatex is available in your PATH)
    subprocess.run(['pdflatex', tex_file])

def convert_pdf_to_png(pdf_file, output_file='output.png'):
    # Convert PDF to PNG (make sure pdftoppm is available in your PATH)
    subprocess.run(['pdftoppm', '-png', '-singlefile', pdf_file, output_file.split('.')[0]])

def main():
    latex_content = r"""
\documentclass[border=8pt, multi=tikz]{standalone}
\usepackage{layers}  % Required for layers functionalities
\usepackage{connection}  % Required for arrows

\definecolor{copper}{rgb}{0.72, 0.45, 0.2}
\definecolor{citrine}{rgb}{0.89, 0.82, 0.04}

\begin{document}
\begin{tikzpicture}
    % Define nodes and connections here...
\end{tikzpicture}
\end{document}
    """
    # Save the LaTeX content to a .tex file
    tex_filename = 'network_diagram.tex'
    pdf_filename = 'network_diagram.pdf'
    png_filename = 'network_diagram.png'

    create_latex_file(latex_content, tex_filename)
    compile_latex_to_pdf(tex_filename)
    convert_pdf_to_png(pdf_filename, png_filename)

if __name__ == "__main__":
    tex_filename = 'network_diagram.tex'
    pdf_filename = 'network_diagram.pdf'
    png_filename = 'network_diagram.png'
    
    # create_latex_file(latex_content, tex_filename)
    compile_latex_to_pdf(tex_filename)
    convert_pdf_to_png(pdf_filename, png_filename)