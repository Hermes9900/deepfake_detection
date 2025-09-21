# Use the official lightweight Nginx image as the base
FROM nginx:alpine

# Remove the default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom Nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/

# Copy our web page into the directory Nginx serves from
COPY index.html /usr/share/nginx/html

# Best practice: Explicitly document the port the container listens on
EXPOSE 80