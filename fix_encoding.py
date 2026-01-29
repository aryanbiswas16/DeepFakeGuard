import os

def convert_to_utf8(filename):
    try:
        with open(filename, 'rb') as f:
            content = f.read()
            
        # Check for null bytes which usually indicates UTF-16
        if b'\x00' in content:
            print(f"Fixing encoding for {filename}")
            # Try decoding as utf-16
            try:
                decoded = content.decode('utf-16')
                # Write back as utf-8
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(decoded)
                print(f"Fixed {filename}")
            except UnicodeError:
                print(f"Could not decode {filename} as utf-16")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            convert_to_utf8(os.path.join(root, file))
print("Encoding fix complete.")
