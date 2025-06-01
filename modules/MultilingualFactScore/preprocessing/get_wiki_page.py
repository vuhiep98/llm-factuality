 
# Import Module
from bs4 import *
import requests
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wiki_link',
                    type=str,
                    default="https://bn.wikipedia.org/wiki/%E0%A6%A1%E0%A7%8B%E0%A6%A8%E0%A6%BE%E0%A6%B2%E0%A7%8D%E0%A6%A1_%E0%A6%9F%E0%A7%8D%E0%A6%B0%E0%A6%BE%E0%A6%AE%E0%A7%8D%E0%A6%AA")
parser.add_argument('--output_file',
                    type=str,
                    default="")
args = parser.parse_args()
# Given URL
url = args.wiki_link
 
# Fetch URL Content
r = requests.get(url)
 
# Get body content
soup = BeautifulSoup(r.text,'html.parser').select('body')[0]
 
# Initialize variable
paragraphs = []
images = []
link = []
heading = []
remaining_content = []
 
# Iterate through all tags
for tag in soup.find_all():
     
    # Check each tag name
    # For Paragraph use p tag
    if tag.name=="p":
       
        # use text for fetch the content inside p tag
        paragraphs.append(tag.text)
         
    # For Image use img tag
    elif tag.name=="img":
       
        # Add url and Image source URL
        images.append(url+tag['src'])
         
    # For Anchor use a tag
    elif tag.name=="a":
       
        # convert into string and then check href 
        # available in tag or not
        if "href" in str(tag):
           
          # In href, there might be possible url is not there
          # if url is not there
            if "https://en.wikipedia.org/w/" not in str(tag['href']):
                link.append(url+tag['href'])
            else:
                link.append(tag['href'])
                 
    # Similarly check for heading 
    # Six types of heading are there (H1, H2, H3, H4, H5, H6)
    # check each tag and fetch text
    elif "h" in tag.name:
        if "h1"==tag.name:
            heading.append(tag.text)
        elif "h2"==tag.name:
            heading.append(tag.text)
        elif "h3"==tag.name:
            heading.append(tag.text)
        elif "h4"==tag.name:
            heading.append(tag.text)
        elif "h5"==tag.name:
            heading.append(tag.text)
        else:
            heading.append(tag.text)
             
    # Remain content will store here
    else:
        remaining_content.append(tag.text)
output = "".join(paragraphs)
with open(args.output_file, 'w') as f:
    f.write(output)     
# print("".join(paragraphs))