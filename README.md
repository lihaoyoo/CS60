# CS60

The  pdfFiles directory contains the implementation code resource files for pdf content extractor.

The Spanbert directory contains the files and saved models for training and evaluating Spanbert model.

The structure for our entity and relation joint model (SpanBERT) is as following:

interface: Project container.

manage.py: Management tool for the Django project.

interface/ __init__.py: An empty file telling Python that the directory is a Python package.

interface/settings.py: Django settings for the entire Django project.

interface/urls.py: The URL declaration for the Django project and a “directory” for the website driven by Django. It defines the mapping of URL to views.

interface/wsgi.py: An entry of web server that is compatible with a WSGI, which can run our project.

interface/run_spanbert.py: This file stores the specific business layer logic, which includes: extraction of entity, 
processing of the dataset, invoking of the prediction relation model, building of the network graph and returning the final results to the template for displaying them on the web.

templates: The directory of template file.

templates/post.html: Html template which describes how the design of this page is. Receiving the data returned by the view module and displays it on the web interface in conjunction with the html language.

Spanbert: Spanbert model.

Spanbert/code/run_tacred_1.py: the main program of relation extraction.
