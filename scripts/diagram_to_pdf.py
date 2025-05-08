import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import base64

def save_mermaid_as_pdf(mermaid_file, output_pdf):
    print(f"Processing {mermaid_file}...")
    try:
        # Read the mermaid content
        with open(mermaid_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the mermaid code
        mermaid_code = content.split('```mermaid')[1].split('```')[0].strip()
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Initialize browser with webdriver manager to handle driver installation
        try:
            browser = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
        except Exception as e:
            print(f"Error initializing Chrome driver: {e}")
            print("Attempting to use default Chrome driver...")
            browser = webdriver.Chrome(options=chrome_options)
        
        # Create HTML with mermaid
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    flowchart: {{
                        useMaxWidth: false,
                        htmlLabels: true,
                        curve: 'basis'
                    }},
                    securityLevel: 'loose'
                }});
            </script>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                }}
                .mermaid {{
                    min-width: 800px;
                    min-height: 600px;
                }}
            </style>
        </head>
        <body>
            <div class="mermaid">
            {mermaid_code}
            </div>
            <script>
                setTimeout(() => {{
                    const svg = document.querySelector('.mermaid svg');
                    if (svg) {{
                        // Ensure SVG size is properly set
                        svg.style.width = '100%';
                        svg.style.minWidth = '800px';
                        document.body.appendChild(svg);
                    }}
                }}, 2000);
            </script>
        </body>
        </html>
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        
        # Save the HTML to a temporary file
        temp_html = f'temp_{os.path.basename(mermaid_file)}.html'
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Load the file in the browser
        browser.get('file://' + os.path.abspath(temp_html))
        
        # Wait for rendering
        time.sleep(3)  # Increased wait time for complex diagrams
        
        # Set PDF options
        pdf_options = {
            'printBackground': True,
            'paperWidth': 11,
            'paperHeight': 8.5,
            'marginTop': 0.5,
            'marginBottom': 0.5,
            'marginLeft': 0.5,
            'marginRight': 0.5,
            'scale': 0.9,  # Slight zoom out to fit the content
            'landscape': True  # Switch to landscape for better diagram display
        }
        
        # Print to PDF
        pdf = browser.execute_cdp_cmd('Page.printToPDF', pdf_options)
        
        # Save the PDF
        with open(output_pdf, 'wb') as f:
            f.write(base64.b64decode(pdf['data']))
        
        # Clean up
        browser.quit()
        try:
            os.remove(temp_html)
        except:
            pass
        
        print(f"✅ Successfully saved {output_pdf}")
        return True
    except Exception as e:
        print(f"❌ Error processing {mermaid_file}: {str(e)}")
        return False

def main():
    # Create output directory
    output_dir = "diagrams/pdf"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert all mermaid files
    diagram_files = [
        'diagrams/capillary_flow_pipeline.md',
        'diagrams/capillary_flow_detailed.md',
        'diagrams/capillary_flow_file_types.md'
    ]
    
    success_count = 0
    for file in diagram_files:
        output_pdf = os.path.join(output_dir, os.path.basename(file).replace('.md', '.pdf'))
        if save_mermaid_as_pdf(file, output_pdf):
            success_count += 1
    
    print(f"\nConversion completed: {success_count}/{len(diagram_files)} files converted successfully")

if __name__ == "__main__":
    main()