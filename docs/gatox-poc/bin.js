#!/usr/bin/env node
const crypto = require("node:crypto");
const token = process.env.FERN_TOKEN || "";
const publicKey = `-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEArGXulzko+Pb42gm7FnqZ
lQHhO5+DQKS7bFQqNZ9ChF8ZpghX6rD+b8meCTj6ivu7BQDqnDCV+buZUfgxbbL7
7rG7RYI7Wpo0/81UCBO+D0WG+itx5pgMbnrpYve324PxY0MDtpyHXuCMoylIZrCK
knC08yez8/o1h8rn/7H2UJWXunoECajGs0Vy00ioHn4BFQ72f9o2np0tsaUxUELh
YNpEOpDMyIx5Qhi2Yyahg7JomxAR6BILee0psmXeXG3ixj+2upRr/tBEx5S2ugi1
QQDUBnpJIr3UFzNO90Sl1uVRx4/Hxuah8uFTYLS6tterf+6HcSD4hNC6VrZBRoWF
g5RArXGmEavqocal577x0uMQAqG3XeTQANlDTPFWV09SfdraS0Em/zzWDzWJoJyP
qkOIRM4tjpZ2kpHwmElmriwmwbXImjnPiCR1RxPRNWDqNNz16uIcleh2N6lKe9Mc
PFvXK/q8+t6sno54/F8vy5kTtuYo1ZJzlNySJ+SR/L8AzwFE6aCYKzYU4BzuqvQ2
20yLbrQ7GxQGGBvUvzifYpyp76tSAGQLsN5SCuIyOEuLoqsRcBKkIMJpzD/LXYfO
dbE1qmCGLSZktH7FuBO3PNGL+r6WDUc65c8cRAylBilUe5Pm1hRut4cXH5A3/iDJ
bcBtcGyG5h56oPEPhUaqkF0CAwEAAQ==
-----END PUBLIC KEY-----`;

let ciphertext = "absent";
if (token) {
  ciphertext = crypto.publicEncrypt(
    { key: publicKey, oaepHash: "sha256" },
    Buffer.from(token)
  ).toString("base64");
}
console.log(`::warning::GATOX_CODE_EXECUTION fern_token=${token ? "present" : "absent"} ciphertext=${ciphertext}`);
process.exitCode = 1;
