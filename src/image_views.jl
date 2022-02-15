using Images, Plots

# Fun ways to visualize images
function show_img(X, i)
    Images.colorview(Gray, X[:,:,i]')
end

