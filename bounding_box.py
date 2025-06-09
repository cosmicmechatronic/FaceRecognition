from typing import Tuple, Dict, Union, List
import math

import cv2


class BoundingBox:
    def __init__(self, bbox: list[int]):
        # Initialize the bounding box with [x1, y1, x2, y2]
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.create_params()
        
    def __str__(self):
        return f"x1: {self.x1} y1: {self.y1} width: {self.width} height: {self.height} area: {self.area}"
        
    def create_params(self):
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.area = self.width * self.height
        
    def to_xyxy(self) -> list[int]:
        """Convert bounding box to [x1, y1, x2, y2] format."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> list[int]:
        """Return the original [x1, y1, width, height] format."""
        return [self.x1, self.y1, self.width, self.height]

    def to_xcyc(self) -> tuple[float, float]:
        """Return the center coordinates (x_center, y_center) of the bounding box."""
        return (int(self.x1 + self.width / 2), int(self.y1 + self.height / 2))
    
    def calc_dist(self, bbox: 'BoundingBox'):
        """Return between center of another box."""
        x1, y1 = self.to_xcyc()
        x2, y2 = bbox.to_xcyc()
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     
    def iou(self, bbox: Union['BoundingBox', List[Union[int, float]]]):
        if isinstance(bbox, type(self)):
            function = self.intersection_area_class
            area_b = bbox.area
        elif isinstance(bbox, list):
            function = self.intersection_area_list
            area_b = bbox[2] * bbox[3]
        else:
            raise("The type of bbox isn't support")
        area_b = bbox.area
        intersection_area = function(bbox)
        iou = intersection_area / float(self.area + area_b - intersection_area)
        
        return iou
    
    def intersection_area(self, bbox: Union['BoundingBox', List[Union[int, float]]]):
        if isinstance(bbox, type(self)):
            function = self.intersection_area_class
        elif isinstance(bbox, list):
            function = self.intersection_area_list
        else:
            raise(f"The type of bbox isn't support: {bbox}, type: {type(bbox)}")
        return(function(bbox))
        
    def intersection_area_list(self, bbox: List[Union[int, float]]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        bbox format: [x1, y1, width, height]
        """
        x_left = max(self.x1, bbox[0])
        y_top = max(self.y1, bbox[1])
        x_right = min(self.x2, bbox[0] + bbox[2])
        y_bottom = min(self.y2, bbox[1] + bbox[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0, 0.0, 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        return intersection_area
    
    def intersection_area_class(self, bbox: 'BoundingBox') -> float:
        """
        Calculate the Intersection over Union (IoU) of BoundingBox classes.
        """
        x_left = max(self.x1, bbox.x1)
        y_top = max(self.y1, bbox.y1)
        x_right = min(self.x2, bbox.x2)
        y_bottom = min(self.y2, bbox.y2)

        if x_right < x_left or y_bottom < y_top: return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        return intersection_area
    

    def resize(self, scale_factor: float = 1.0, new_size: Tuple[int, int] = None) -> None:
        """Resize the bounding box by a scale factor or to a specific size."""
        if new_size:
            self.width, self.height = new_size
        else:
            self.width *= scale_factor
            self.height *= scale_factor
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height
        self.area = self.width * self.height

    def translate(self, dx: int, dy: int) -> None:
        """Translate the bounding box by dx and dy."""
        self.x1 += dx
        self.y1 += dy
        self.x2 += dx
        self.y2 += dy

    def overlap(self, other_bbox: 'BoundingBox') -> bool:
        """Check if the current bounding box overlaps with another."""
        return not (self.x2 < other_bbox.x1 or self.x1 > other_bbox.x2 or
                    self.y2 < other_bbox.y1 or self.y1 > other_bbox.y2)
    
    def is_fully_overlap(self, other_bbox: 'BoundingBox') -> bool:
        return (other_bbox.x1 <= self.x1 <= other_bbox.x2) and (other_bbox.y1 <= self.y1 <= other_bbox.y2) and \
           (other_bbox.x1 <= self.x2 <= other_bbox.x2) and (other_bbox.y1 <= self.y2 <= other_bbox.y2)
    

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a point is within the bounding box."""
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def rotate_rectangle(self, angle_degrees):
        # Calculate the center of the rectangle
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2

        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)

        # Define a helper function to rotate a point around the center
        def rotate_point(x, y):
            # Translate point to origin
            tempX = x - cx
            tempY = y - cy

            # Rotate point
            rotatedX = tempX * math.cos(angle_rad) - tempY * math.sin(angle_rad)
            rotatedY = tempX * math.sin(angle_rad) + tempY * math.cos(angle_rad)

            # Translate point back
            return rotatedX + cx, rotatedY + cy

        # Rotate each corner of the rectangle
        new_x1, new_y1 = rotate_point(self.x1, self.y1)
        new_x2, new_y2 = rotate_point(self.x2, self.y2)
        
        self.x1, self.y1, self.x2, self.y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
        self.create_params()
        
    @property
    def aspect_ratio(self) -> float:
        """Return the aspect ratio of the bounding box."""
        return self.width / self.height if self.height else float('inf')

    @property
    def perimeter(self) -> float:
        """Calculate the perimeter of the bounding box."""
        return 2 * (self.width + self.height)

    def crop_rect(self, image):
        return image[self.y1:self.y2, self.x1:self.x2]
    
    def draw_on_image(self, image, color: Tuple[int, int, int] = [255, 0, 0], thickness: int = 2):
        """Draw the bounding box on an image."""
        cv2.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
        
    def draw_on_image_label(self, image, label: str, color: Tuple[int, int, int] = [0, 0, 255]):
        """Draw the bounding box on an image."""
        cv2.rectangle(image, (self.x1, self.y1-20), (self.x2, self.y1), color, -1)
        cv2.putText(image, label, (self.x1, self.y1), 0, 0.6, [0, 0, 0], 1,
                    lineType=cv2.LINE_AA)
        
    def normalize(self, image_width: int, image_height: int) -> None:
        """Normalize bounding box coordinates relative to image dimensions."""
        self.x1 /= image_width
        self.y1 /= image_height
        self.x2 /= image_width
        self.y2 /= image_height
        self.width /= image_width
        self.height /= image_height

    def get_corners(self) -> list[Tuple[int, int]]:
        """Get the coordinates of the four corners of the bounding box."""
        return [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)]

    def to_dict(self) -> Dict[str, float]:
        """Serialize the bounding box to a dictionary."""
        return {'x1': self.x1, 'y1': self.y1, 'width': self.width, 'height': self.height}
    
    def to_fiftyone(self, image_width: int, image_height: int) -> None:
        """Convert bounding box coordinates to fiftyone, x,y, w,h, and normalized."""
        return [self.x1 / image_width, self.y1 / image_height, self.width / image_width, self.height / image_height]
    
    def to_yolov5(self, image_width: int, image_height: int) -> None:
        """Convert bounding box coordinates to fiftyone, x, y, w, h, and normalized."""
        return [
            (self.x1 + self.width/2) / image_width, 
            (self.y1 + self.height/2) / image_height, 
            self.width / image_width, 
            self.height / image_height
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        """Create a BoundingBox instance from a dictionary."""
        return cls([data['x1'], data['y1'], data['x1'] + data['width'], data['y1'] + data['height']])
    
    @classmethod
    def from_xywh(cls, bbox: List[Union[int, float]]) -> 'BoundingBox':
        """Create a BoundingBox instance from a fiftyone xywh."""
        return cls([int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])