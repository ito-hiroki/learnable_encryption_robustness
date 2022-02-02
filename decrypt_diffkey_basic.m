input_dir = './encrypted_images/etc/test/same_key/';
output_dir = './attacked_images/fr/etc/test/same_key/';

if not(exist(output_dir,'dir'))
    mkdir(output_dir)
end

input_info_list = dir(fullfile(input_dir, '*.png'));
input_img_list = fullfile(input_dir, {input_info_list.name});

for img_path = input_img_list
    
    [folder, name, ext] = fileparts(char(img_path));
    enc = imread(char(img_path));
    size_x = size(enc, 1);
    size_y = size(enc, 2);

    dec = zeros(size_x, size_y, 3, 'uint8');

    for i = 1:size_x
        for j = 1:size_y
            for rgb = 1:3
                val = enc(i, j, rgb);
                if val <= 127
                    dec(i, j, rgb) = val;
                else
                    dec(i, j, rgb) = bitxor(val, 255);
                end
            end
        end
    end
    
    imwrite(dec, append(output_dir, name, ext));
end