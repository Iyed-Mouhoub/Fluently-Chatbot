import PyPDF2
import os

def convert_pdf_to_txt(pdf_path, output_folder):
    """Convert a single PDF to TXT file"""
    try:
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Extract text
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"--- Page {page_num + 1} ---\n"
                text += page_text + "\n\n"
        
        # Save to TXT file
        output_path = os.path.join(output_folder, f"{filename}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✅ Converted: {filename}.pdf → {filename}.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {pdf_path}: {e}")
        return False

def convert_all_pdfs(input_folder, output_folder):
    """Convert all PDFs in a folder to TXT files"""
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_folder}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    print("🔄 Converting PDFs to TXT files...\n")
    
    success_count = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        if convert_pdf_to_txt(pdf_path, output_folder):
            success_count += 1
    
    print(f"\n✨ Conversion complete! {success_count}/{len(pdf_files)} files converted")
    print(f"📂 TXT files saved in: {output_folder}")

if __name__ == "__main__":
    # Configuration
    PDF_FOLDER = "pdfs"  # Put your PDF files here
    OUTPUT_FOLDER = "course_materials"  # TXT files will be saved here
    
    print("🇫🇷 French Course PDF Converter")
    print("=" * 40)
    
    # Check if PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder '{PDF_FOLDER}' not found!")
        print(f"📝 Create a folder called '{PDF_FOLDER}' and put your 20 PDF files there")
    else:
        convert_all_pdfs(PDF_FOLDER, OUTPUT_FOLDER)
        
        # Show preview of first file
        txt_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.txt')]
        if txt_files:
            print(f"\n👀 Preview of {txt_files[0]}:")
            print("-" * 40)
            with open(os.path.join(OUTPUT_FOLDER, txt_files[0]), 'r', encoding='utf-8') as f:
                preview = f.read()[:500] + "..." if len(f.read()) > 500 else f.read()
                print(preview)