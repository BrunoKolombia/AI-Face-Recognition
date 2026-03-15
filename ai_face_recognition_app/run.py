import sys
import os

src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from src.main import main

if __name__ == "__main__":
    print("Starting Face Recognition App...")
    print("Press Ctrl+C to exit")

    main()