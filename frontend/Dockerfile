FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
ARG NGINX_USER=admin
ARG NGINX_PASSWORD
RUN printf "%s:%s\n" "$NGINX_USER" "$(openssl passwd -apr1 "$NGINX_PASSWORD")" \
    > /etc/nginx/.htpasswd
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
