import json

import cv2
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from PIL import Image
from datetime import datetime

def board_to_json(board, dict_name):
    marker_data = {"dictionary": dict_name, "markers": []}

    for i, corner in enumerate(board.getObjPoints()):
        marker_data["markers"].append(
            {
                "id": int(board.getIds()[i]),  # Marker ID
                "corners": corner.tolist(),  # Convert NumPy array to list
            }
        )

    return marker_data

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    num_markers_x = 3
    num_markers_y = 3
    marker_separation = 0.004
    marker_length = (0.1 - marker_separation*(num_markers_x + 1))/num_markers_x
    start_id = 200

    board = cv2.aruco.GridBoard(
        (
            num_markers_x,  # Number of markers in the X direction.
            num_markers_y,
        ),  # Number of markers in the Y direction.
        marker_length,  # Length of the marker side.
        marker_separation,  # Length of the marker separation.
        dictionary,  # The dictionary of the markers.
        np.array(
            [i + start_id for i in range(num_markers_x * num_markers_y)]
        ),  # (optional) Ids of all the markers (X*Y markers).
    )

    # Compute board size in mm
    board_width_mm = (
        num_markers_x * marker_length + (num_markers_x - 1) * marker_separation
    ) * 1000
    board_height_mm = (
        num_markers_y * marker_length + (num_markers_y - 1) * marker_separation
    ) * 1000

    print(f"Board size: {board_width_mm} mm x {board_height_mm} mm")

    # Create high-res image
    dpi = 300
    pixels_per_mm = dpi / 25.4
    img_width_px = int(board_width_mm * pixels_per_mm)
    img_height_px = int(board_height_mm * pixels_per_mm)

    img = board.generateImage((img_width_px, img_height_px), marginSize=0, borderBits=1)

    # Convert and save as PNG temporarily
    img_pil = Image.fromarray(img)
    tmp_png = "temp_board.png"
    img_pil.save(tmp_png)

    # Create PDF with correct physical size
    date_str = datetime.now().strftime("%Y-%m-%d")
    marker_info = f"DICT_4X4_250_{num_markers_x}x{num_markers_y}_{marker_length*1000:.1f}mm"
    title = f"Aruco board {marker_info} {date_str}"
    pdf_filename = f"aruco_board_{marker_info}_{date_str}.pdf"

    border_mm = marker_separation * 1000
    page_width = (board_width_mm + 2 * border_mm) * mm
    page_height = (board_height_mm + 2 * border_mm) * mm
    print(f"Size with borders: {page_width / mm} mm x {page_height / mm} mm")

    c = canvas.Canvas(pdf_filename, pagesize=(page_width, page_height))
    c.setTitle(title)

    x = border_mm * mm
    y = border_mm * mm

    c.drawImage(tmp_png, x, y, width=board_width_mm * mm, height=board_height_mm * mm)
    c.setLineWidth(0.5)
    c.rect(0, 0, page_width, page_height, stroke=1, fill=0)

    c.showPage()
    c.save()

    save_path = f"aruco_board_{marker_info}_{date_str}.json"
    serialized_board = board_to_json(board, "DICT_4X4_250")
    
    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(serialized_board, f, indent=4)

if __name__ == "__main__":
    main()