```
\documentclass[10pt, letterpaper]{article}

% Packages:
\usepackage[
    ignoreheadfoot, % set margins without considering header and footer
    top=2 cm, % seperation between body and page edge from the top
    bottom=2 cm, % seperation between body and page edge from the bottom
    left=2 cm, % seperation between body and page edge from the left
    right=2 cm, % seperation between body and page edge from the right
    footskip=1.0 cm, % seperation between body and footer
    % showframe % for debugging 
]{geometry} % for adjusting page geometry
\usepackage{titlesec} % for customizing section titles
\usepackage{tabularx} % for making tables with fixed width columns
\usepackage{array} % tabularx requires this
\usepackage[dvipsnames]{xcolor} % for coloring text
\definecolor{primaryColor}{RGB}{0, 79, 144} % define primary color
\usepackage{enumitem} % for customizing lists
\usepackage{fontawesome5} % for using icons
\usepackage{amsmath} % for math
\usepackage[
    pdftitle={DHARUN M R DS RESUME},
    pdfauthor={DHARUN M R},
    pdfcreator={LaTeX with RenderCV},
    colorlinks=true,
    urlcolor=primaryColor
]{hyperref} % for links, metadata and bookmarks
\usepackage[pscoord]{eso-pic} % for floating text on the page
\usepackage{calc} % for calculating lengths
\usepackage{bookmark} % for bookmarks
\usepackage{lastpage} % for getting the total number of pages
\usepackage{changepage} % for one column entries (adjustwidth environment)
\usepackage{paracol} % for two and three column entries
\usepackage{ifthen} % for conditional statements
\usepackage{needspace} % for avoiding page brake right after the section title
\usepackage{iftex} % check if engine is pdflatex, xetex or luatex

% Ensure that generate pdf is machine readable/ATS parsable:
\ifPDFTeX
    \input{glyphtounicode}
    \pdfgentounicode=1
    % \usepackage[T1]{fontenc} % this breaks sb2nov
    \usepackage[utf8]{inputenc}
    \usepackage{lmodern}
\fi

% Some settings:
\AtBeginEnvironment{adjustwidth}{\partopsep0pt} % remove space before adjustwidth environment
\pagestyle{empty} % no header or footer
\setcounter{secnumdepth}{0} % no section numbering
\setlength{\parindent}{0pt} % no indentation
\setlength{\topskip}{0pt} % no top skip
\setlength{\columnsep}{0cm} % set column seperation
\makeatletter
\let\ps@customFooterStyle\ps@plain % Copy the plain style to customFooterStyle
% \patchcmd{\ps@customFooterStyle}{\thepage}{
%     \color{gray}\textit{\small John Doe - Page \thepage{} of \pageref*{LastPage}}
% }{}{} % replace number by desired string
\makeatother
% \pagestyle{customFooterStyle}

\titleformat{\section}{\needspace{4\baselineskip}\bfseries\large}{}{0pt}{}[\vspace{1pt}\titlerule]

\titlespacing{\section}{
    % left space:
    -1pt
}{
    % top space:
    0.3 cm
}{
    % bottom space:
    0.2 cm
} % section title spacing

\renewcommand\labelitemi{$\circ$} % custom bullet points
\newenvironment{highlights}{
    \begin{itemize}[
        topsep=0.10 cm,
        parsep=0.10 cm,
        partopsep=0pt,
        itemsep=0pt,
        leftmargin=0.4 cm + 10pt
    ]
}{
    \end{itemize}
} % new environment for highlights

\newenvironment{highlightsforbulletentries}{
    \begin{itemize}[
        topsep=0.10 cm,
        parsep=0.10 cm,
        partopsep=0pt,
        itemsep=0pt,
        leftmargin=10pt
    ]
}{
    \end{itemize}
} % new environment for highlights for bullet entries

\newenvironment{onecolentry}{
    \begin{adjustwidth}{
        0.2 cm + 0.00001 cm
    }{
        0.2 cm + 0.00001 cm
    }
}{
    \end{adjustwidth}
} % new environment for one column entries

\newenvironment{twocolentry}[2][]{
    \onecolentry
    \def\secondColumn{#2}
    \setcolumnwidth{\fill, 4.5 cm}
    \begin{paracol}{2}
}{
    \switchcolumn \raggedleft \secondColumn
    \end{paracol}
    \endonecolentry
} % new environment for two column entries

\newenvironment{header}{
    \setlength{\topsep}{0pt}\par\kern\topsep\centering\linespread{1.5}
}{
    \par\kern\topsep
} % new environment for the header

% save the original href command in a new command:
\let\hrefWithoutArrow\href

% new command for external links:
\renewcommand{\href}[2]{\hrefWithoutArrow{#1}{\ifthenelse{\equal{#2}{}}{ }{#2 }\raisebox{.15ex}{\footnotesize \faExternalLink*}}}

\begin{document}
    \newcommand{\AND}{\unskip
        \cleaders\copy\ANDbox\hskip\wd\ANDbox
        \ignorespaces
    }
    \newsavebox\ANDbox
    \sbox\ANDbox{}

    
    \begin{header}
        \textbf{\fontsize{24 pt}{24 pt}\selectfont DHARUN M R}

        \vspace{0.3 cm}

        \normalsize
        \mbox{{\color{black}\footnotesize\faMapMarker*}\hspace*{0.13cm}Tirunelveli, Tamilnadu}%
        \kern 0.25 cm%
        \AND%
        \kern 0.25 cm%
        \mbox{\hrefWithoutArrow{mailto:dharunmr.ind@gmail.com}{\color{black}{\footnotesize\faEnvelope[regular]}\hspace*{0.13cm}dharunmr.ind@gmail.com}}%
        \kern 0.25 cm%
        \AND%
        \kern 0.25 cm%
        \mbox{\hrefWithoutArrow{tel:+91-7695953002}{\color{black}{\footnotesize\faPhone*}\hspace*{0.13cm}7695953002}}%
        \kern 0.25 cm%
        \AND%
        \kern 0.25 cm%
        \AND%
        \kern 0.25 cm%
        \mbox{\hrefWithoutArrow{https://linkedin.com/in/dharun-m-r-145364312}{\color{black}{\footnotesize\faLinkedinIn}\hspace*{0.13cm}dharun-m-r-145364312}}%
        \kern 0.25 cm%
        \AND%
        \kern 0.25 cm%
        \mbox{\hrefWithoutArrow{https://github.com/dharun-gitspace}{\color{black}{\footnotesize\faGithub}\hspace*{0.13cm}dharun-gitspace}}%
    \end{header}

    \vspace{0.3 cm - 0.3 cm}


    \section{Profile}

        \begin{onecolentry}
          
An experienced and analytical student with a strong foundation in data science and machine learning. Gained hands-on experience through diverse projects, applying statistical analysis, machine learning models, and data visualization to derive meaningful insights. A quick learner with a keen interest in data-driven problem-solving, always eager to explore new challenges and enhance practical knowledge through real-world data applications.
        \end{onecolentry}

        \vspace{0.2 cm}
    
    % \section{Quick Guide}

    % \begin{onecolentry}
    %     \begin{highlightsforbulletentries}

    %     \item Each section title is arbitrary and each section contains a list of entries.

    %     \item There are 7 unique entry types: \textit{BulletEntry}, \textit{TextEntry}, \textit{EducationEntry}, \textit{ExperienceEntry}, \textit{NormalEntry}, \textit{PublicationEntry}, and \textit{OneLineEntry}.

    %     \item Select a section title, pick an entry type, and start writing your section!

    %     \item \href{https://docs.rendercv.com/user_guide/}{Here}, you can find a comprehensive user guide for RenderCV.

    %     \end{highlightsforbulletentries}
    % \end{onecolentry}

    \section{Education}
        
        \begin{twocolentry}{
                        
        \textit{2022 - present}}
            \textbf{Master of Science in Software Systems Integrated} 

            \text{Coimbatore Institute of Technology, Coimbatore, India}
        \end{twocolentry}

        \vspace{0.10 cm}
        
        \begin{onecolentry}
            \begin{highlights}
                \item CGPA: 7.23 
            \end{highlights}
        \end{onecolentry}
        
        \vspace{0.10 cm}
        
        \begin{twocolentry}{    
        \textit{2022}}
            \textbf{Higher Secondary - 12 th grade}

            \text{Kings Matric higher Secondary School, Tirunelveli, India}
        \end{twocolentry}

        \vspace{0.10 cm}

        \begin{onecolentry}
            \begin{highlights}
                \item AGGREGATE: 88.33 
            \end{highlights}
        \end{onecolentry}
   
    \section{Skills}
        \begin{tabular}{ p{7cm} p{7cm} }
            \textbf{Programming Languages} & \textbf{Tools} \\
            Python(intermediate), C, Java, HTML, JS(intermediate), SQL, NOSQL & OracleDB, MySQL, Spring Initializr, Git, GitHub, Linux-Arch\\
            
            \vspace{0.10 cm}

    \end{tabular}

        \hspace{0.25cm}\textbf{Frameworks}
        \begin{onecolentry}
            \hspace{0.1 cm}CrewAI, Rapidminer(basic), weka(basic), ReactNative(intermediate), Django(basic), Spring Boot
        \end{onecolentry}
    
    \section{Certificates and Achievements}     
            \begin{tabular}{ p{7cm} p{7cm} }
                \textbf{Microsoft} & \textbf{CIT} \\
                Microsoft LLM workshop & CIT Spark Grant Hackathon 2024 Finalist \\
            \end{tabular}
        
            \vspace{0.10 cm}
            \begin{tabular}{ p{7cm} p{7cm} }
                \textbf{Google Developer Students Club} & \textbf{Hack\$day} \\
                GDSC Summer Hackathon 2023 & 0x Day Hackathon 2024 Level-2 2024 Finalist \\
            \end{tabular}
    
    \section{Projects}

        \vspace{0.20 cm}
        
        \begin{twocolentry}{
        \textit{\href{https://github.com/dharun-gitspace/hackday}{}}}
            \textbf{IAP}
        \end{twocolentry}

        \vspace{0.10 cm}
        \begin{onecolentry}
            \begin{highlights}
                \item The Intelligent Assessment Platform is a web-based application designed for teachers and students. It leverages AI to generate topic-wise MCQs, facilitates adaptive testing, and identifies student weak areas for retesting. The platform delivers an efficient way for teachers to manage assessments and for students to improve their learning outcomes.
                \item \textbf{Tools Used :} ReactJS, Python, Java Spring Boot, MongoDB
                \item \textbf{AI Tools Used :}  CrewAI, Ollama, Lama2 (7B)
            \end{highlights}
        \end{onecolentry}

        \vspace{0.20 cm}
        
        \vspace{0.30 cm}

        \begin{twocolentry}{
        \textit{\href{}{}}}
            \textbf{SMP NET}
        \end{twocolentry}
        \vspace{0.10 cm}
        \begin{onecolentry}
            \begin{highlights}
                \item Developed a stock market prediction system that leverages a Recurrent Neural Network (RNN) to analyze historical stock price data and forecast future trends. By utilizing time series analysis, the system aims to provide accurate stock price predictions, helping traders and investors make data-driven decisions.
                \item \textbf{Tools Used :} Python, keras model, Tensorflow, Yfinance
            \end{highlights}
        \end{onecolentry}

\end{document}
```