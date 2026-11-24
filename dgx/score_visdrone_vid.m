% Invoke the official VisDrone VID toolkit with explicit paths and clip list.
args = argv();
root = args{1};
resultDir = args{2};
listFile = args{3};
outFile = args{4};
pkg load image;
addpath(args{5});
fid = fopen(listFile);
names = textscan(fid, '%s');
fclose(fid);
names = names{1};
allgt = cell(1, numel(names));
alldet = cell(1, numel(names));
for i = 1:numel(names)
    name = names{i};
    gt = dlmread(fullfile(root, 'annotations', [name '.txt']), ',');
    file = fullfile(resultDir, [name '.txt']);
    info = dir(file);
    det = zeros(0, 10);
    if info.bytes > 0
        det = dlmread(file, ',');
    end
    frames = dir(fullfile(root, 'sequences', name, '*.jpg'));
    image = imfinfo(fullfile(root, 'sequences', name, frames(1).name));
    [gt, det] = dropObjectsInIgr(gt, det, image.Height, image.Width);
    gt(:, 7) = (gt(:, 7) == 0);
    allgt{i} = gt;
    alldet{i} = det;
end
[~, ~, ap, ap50, ap75, ar1, ar10, ar100, ar500] = calcAccuracy(numel(names), allgt, alldet);
fid = fopen(outFile, 'w');
fprintf(fid, '%.10g %.10g %.10g %.10g %.10g %.10g %.10g\n', ap, ap50, ap75, ar1, ar10, ar100, ar500);
fclose(fid);
