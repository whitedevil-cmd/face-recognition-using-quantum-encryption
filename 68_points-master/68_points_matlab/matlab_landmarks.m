Img = imread("F:\68_points\68_points_matlab\test.jpg");
[bboxes,shapes]=face_landmarks(Img);
face=insertObjectAnnotation(Img,'rectangle',bboxes,'face');%���rectangle��
faceLandmarks=insertMarker(face,shapes);                   %���landmarks
imshow(faceLandmarks);